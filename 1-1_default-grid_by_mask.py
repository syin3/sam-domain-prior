"""
Inference with default grid in SAM's transformer implementation; 

Usage:
(1) python 1-1_default_grid_by_mask.py --pts_per_side 32 --sam_siz vit_b \
    --uncertainty_strategy relative_0.1 --domain_name mask2former --world_size 1 --data_dir "../data/mapillary_exm" \
    --data_grp validation --num_img 10 --sam_author hugging_face --pts_per_batch 32 --crop_n_layers 0 \
    --crop_n_points_downscale_factor 1.0 
"""
import sys, os, argparse, time
import numpy as np
import torch
from pathlib import Path

from PIL import Image
import pycocotools.mask as mask_utils
import mmcv

from src import utils, core, metrics, visuals
from src.configs import FB_CONFIG, MMLAB_CONFIG

import torch.distributed as dist
import torch.multiprocessing as mp

import logging
logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("default_grid.log", mode="w"),
    ],
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='SAM choose vote')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--data_dir', help='specify the path of input images')
    parser.add_argument('--data_grp', type=str, default='both', choices=['training', 'validation'], help='group of data to inference')
    parser.add_argument('--num_img', type=int, default=0, help='number of images to process by all nodes')
    parser.add_argument('--sam_siz', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], help='which SAM size to load')
    parser.add_argument('--domain_name', type=str, default='mask2former', choices=['mask2former', 'deeplab', 'ground_truth'], help='which domain seg model to load')
    # parser.add_argument('--domain_root', type=str, help='root to load existing results of domain model')
    parser.add_argument('--sam_author', type=str, choices=['facebook', 'hugging_face'], help='implementation version of SAM by FB or Hugging Face')
    parser.add_argument('--pts_per_side', type=int, default=128, help='number of points per side to generate prompt grid')
    parser.add_argument('--pts_per_batch', type=int, default=64, help='number of points to process per batch')
    parser.add_argument('--crop_n_layers', type=int, default=0, help='number of crop layers')
    parser.add_argument('--crop_n_points_downscale_factor', type=float, default=1.0, help='downscale factor')
    parser.add_argument('--uncertainty_strategy', type=str, default='relative_0.1', help='strategy and threshold for uncertainty sampling')

    args = parser.parse_args()
    return args

def main(rank, root_dir, args):
    """
    size:
        prd.size() - [65, 3000, 4000]
        prd.argmax(dim=0).size() - [3000, 4000]
        prd.argmax(dim=0, keepdim=True).size() - [1, 3000, 4000]
    """
    if args.world_size > 1:
        # PyTorch initializes the distributed communication backend (e.g., NCCL, Gloo)
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    sam_model = core.load_sam(
        author=args.sam_author,
        rank=rank,
        size=args.sam_siz,
        pts_per_side=args.pts_per_side,
        pts_per_batch=args.pts_per_batch,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor
    )

    train_img = [fn_ for fn_ in os.listdir(os.path.join(args.data_dir, args.data_grp, 'images')) if '.jpg' in fn_]
    if args.num_img > 0:
        train_img = train_img[:args.num_img]

    local_img = train_img[(len(train_img) // args.world_size) * rank : (len(train_img) // args.world_size) * (rank + 1)]
    logger.info(f"[Img] {len(local_img):,} image files identified for device {rank}")

    # local_img = ["dejslqZtqGZ53SiXTInqug.jpg"]

    for i, file_name in enumerate(local_img):
        file_path = os.path.join(args.data_dir, args.data_grp, 'images', file_name)
        # argmax class from the mean probability C*H*W -> H*W
        # domain_pred = torch.from_numpy(np.load(os.path.join(args.domain_root, f"results_{args.domain_name}", args.data_grp, 'npy', 'labl', file_name.replace('jpg', 'npy')))).to(torch.int64).to(device)
        if args.domain_name != "ground_truth":
            domain_pred = np.argmax(
                np.load(
                    os.path.join(
                        f"../results/",
                        f"{args.domain_name}", 
                        f"{args.data_grp}", 
                        file_name.replace(".jpg", "_mean.npy")
                    )), 
                axis=0
            )
        else:
            domain_pred = np.array(
                Image.open(os.path.join(
                    "../data/mapillary_exm/validation/labels/",
                    file_name.replace(".jpg", ".png")
                ))
            ).astype(np.int64)

        logger.info(f"[Argmax]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")
        # diff vs. torch.no_grad(): https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099?u=timgianitsos
        with torch.inference_mode():
            masks = core.infer_sam(
                file_path, 
                args.sam_author, 
                sam_model,
                pts_per_side=args.pts_per_side, 
                pts_per_batch=args.pts_per_batch,
                crop_n_layers=args.crop_n_layers, 
                crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
                uncertainty_file="",
            )
            logger.info(f"Number of masks: {len(masks)}")
            logger.info(f"[SAM]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

            # voting
            merged_mask = core.domain_vote_per_sam_mask(
                masks, 
                torch.from_numpy(domain_pred).to(device), 
                uncertainty_np=np.load(
                    os.path.join(
                        "../results/",
                        f"{args.domain_name}" if args.domain_name != "ground_truth" else "mask2former",
                        f"{args.data_grp}",
                        f"{file_name.replace('.jpg', '_mut.npy')}"
                    )), 
                uncertainty_strategy=args.uncertainty_strategy,
                author=args.sam_author,
                rank=rank
            )

            # save composite label and img
            visuals.draw_save_prd(
                    mask=merged_mask,
                    save_mask_dir=os.path.join("../results", root_dir, args.data_grp, 'npy', 'labl', file_name.replace('jpg', 'npy')),
                    img_array=mmcv.imread(file_path, backend='pillow'), # img_array
                    save_img_dir=os.path.join("../results", root_dir, args.data_grp, 'img', file_name),
                    rank=rank,
                    granular='low'
            )
            try:
                logger.info(f"[Success]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

            except Exception as e:
                logger.warning(f"[Failed]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}; due to {e}")

if __name__ == '__main__':
    args = parse_args()

    # root_dir has three groups of keywords
    # (1) domain_name: which model generated domain_pred and 
    # (2) uncertainty_strategy: how uncertainty are utilized to vote
    # (3) sam config: which sam model generated the overlayed masks
    root_dir = f"{args.domain_name}_{args.uncertainty_strategy}_{args.sam_siz[-1]}_{args.pts_per_side}_{args.crop_n_layers}_{args.crop_n_points_downscale_factor}"
    Path("../results", root_dir, args.data_grp, 'img').mkdir(parents=True, exist_ok=True)
    Path("../results", root_dir, args.data_grp, 'npy', 'labl').mkdir(parents=True, exist_ok=True)

    if args.world_size > 1:
        mp.spawn(main,args=(root_dir, args,),nprocs=args.world_size,join=True)
    else:
        main(0, root_dir, args)
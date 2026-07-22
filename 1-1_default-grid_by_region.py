"""
Inference with default grid in SAM's transformer implementation; 

Usage:
(1) python 1-1_default_grid_by_region.py --pts_per_side 32 --sam_siz vit_b \
    --uncertainty_strategy relative_0.1 --domain_name mask2former --world_size 1 --data_dir "../data/mapillary_exm" \
    --data_grp validation --start_img_idx 0 --end_img_idx 10 --sam_author hugging_face --pts_per_batch 32 --crop_n_layers 0 \
    --crop_n_points_downscale_factor 1.0 
"""
import sys, os, argparse, time, traceback
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
import gc

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
    parser.add_argument('--start_img_idx', type=int, default=0, help='start index of images to process by all nodes')
    parser.add_argument('--end_img_idx', type=int, default=0, help='end index of images to process by all nodes')
    parser.add_argument('--sam_siz', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], help='which SAM size to load')
    parser.add_argument('--domain_name', type=str, default='mask2former', choices=['mask2former', 'deeplab', 'ground-truth'], help='which domain seg model to load')
    # parser.add_argument('--domain_root', type=str, help='root to load existing results of domain model')
    parser.add_argument('--sam_author', type=str, choices=['facebook', 'hugging_face'], help='implementation version of SAM by FB or Hugging Face')
    parser.add_argument('--pts_per_side', type=int, default=128, help='number of points per side to generate prompt grid')
    parser.add_argument('--pts_per_batch', type=int, default=64, help='number of points to process per batch')
    parser.add_argument('--crop_n_layers', type=int, default=0, help='number of crop layers')
    parser.add_argument('--crop_n_points_downscale_factor', type=float, default=1.0, help='downscale factor')
    parser.add_argument('--uncertainty_strategy', type=str, default='relative_0.1', help='strategy and threshold for uncertainty sampling')
    parser.add_argument('--overlay_only', type=str, default='False', choices=['True', 'False'], help='only the overlayed mask; do not vote')

    args = parser.parse_args()
    return args

def main(rank, root_dir, args):
    """
    size:
        prd.size() - [65, 3000, 4000]
        prd.argmax(dim=0).size() - [3000, 4000]
        prd.argmax(dim=0, keepdim=True).size() - [1, 3000, 4000]
    """

    # set up
    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    sam_model = None

    untouched_files = []
    train_img = [fn_ for fn_ in os.listdir(os.path.join(args.data_dir, args.data_grp, 'images')) if '.jpg' in fn_]

    train_img = train_img[args.start_img_idx:args.end_img_idx]
    logger.info(f"[Img]: {len(train_img):,} total images")
    
    for file_name in train_img:
        overlayed_mask_dir = os.path.join(
            "../results", 
            file_name.replace(
                ".jpg", 
                f"_overlayed_{args.sam_siz[-1]}_{args.pts_per_side}_{args.crop_n_layers}_{args.crop_n_points_downscale_factor}.npy"
            )
        )
        if args.overlay_only == "False" and all(
            os.path.exists(os.path.join("../results", root_dir, args.data_grp, 'npy', 'labl', file_name.replace('.jpg', f'_{idx}.npy'))) and \
            os.path.exists(os.path.join("../results", root_dir, args.data_grp, 'img', file_name.replace('.jpg', f'_{idx}.jpg'))) 
            for idx in [0, 1]
        ):
            # now want to vote, but results already exist, skip
            ## results are covered and full merge masks, idx=0 and idx=1
            # results exist, no need for any forward pass
            continue
        elif args.overlay_only == "True" and os.path.exists(overlayed_mask_dir):
            continue
        else:
            untouched_files.append(file_name)
    local_img = untouched_files[(len(untouched_files) // args.world_size) * rank : (len(untouched_files) // args.world_size) * (rank + 1)]

    # local_img = ["j2jvTIJhp98QIi3hTa5fHg.jpg"]

    logger.info(f"[Img] {len(local_img):,} image files identified for device {rank}")

    # process images
    for i, file_name in enumerate(local_img):
        if file_name == "tDUfH-CCdIj1QMAWKhK9oA.jpg":
            continue
        file_path = os.path.join(args.data_dir, args.data_grp, 'images', file_name)
        logger.info(f"[Start]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

        # check if overlayed mask exists for the image
        overlayed_mask_dir = os.path.join(
            "../results", 
            file_name.replace(
                ".jpg", 
                f"_overlayed_{args.sam_siz[-1]}_{args.pts_per_side}_{args.crop_n_layers}_{args.crop_n_points_downscale_factor}.npy"
            )
        )

        if not os.path.exists(overlayed_mask_dir):
            if sam_model is None:
                sam_model = core.load_sam(
                    author=args.sam_author,
                    rank=rank,
                    size=args.sam_siz,
                    pts_per_side=args.pts_per_side,
                    pts_per_batch=args.pts_per_batch,
                    crop_n_layers=args.crop_n_layers,
                    crop_n_points_downscale_factor=args.crop_n_points_downscale_factor
                )
            # if overlayed mask does not exist, always SAM forward
            current_pts_per_batch = args.pts_per_batch
            max_retries = 3
            attempt = 0
            success = False
            while attempt < max_retries and not success:
                try:
                    with torch.inference_mode():
                        # Clear GPU cache before each attempt
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        
                        logger.info(f"[Attempt {attempt+1}] Using points_per_batch={current_pts_per_batch}")

                        masks = core.infer_sam(
                            file_path, 
                            args.sam_author, 
                            sam_model,
                            pts_per_side=args.pts_per_side, 
                            pts_per_batch=current_pts_per_batch,
                            crop_n_layers=args.crop_n_layers, 
                            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
                            uncertainty_file="",
                        )
                        success = True
                except Exception as e:
                    logger.error(f"[Error] {file_name} attempt {attempt+1}/{max_retries}: {e}\n[Traceback] Full error trace:\n{traceback.format_exc()}")
                    current_pts_per_batch = max(current_pts_per_batch // 2, 16)  # Minimum 16
                    attempt += 1
            
            if not success:
                logger.error(f"[Failed] {file_name} after {max_retries} attempts")
                continue
            
            # generate overlayed mask
            overlayed_mask_np = core.overlay_sam_masks(
                masks["masks"] if "grid_points" in masks else masks, 
                author=args.sam_author
            )
            np.save(overlayed_mask_dir, overlayed_mask_np)
            logger.info(f"[Overlayed saved]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

        # now overlayed mask exists, return early if not voting
        if args.overlay_only == "True":
            continue
        
        ## load domain pred 
        if args.domain_name != "ground-truth":
            mean_file = os.path.join(
                f"../results/",
                f"{args.domain_name}", 
                f"{args.data_grp}", 
                file_name.replace(".jpg", "_mean.npy") if file_name != "p_abpafjpgfUaYs4kCSF7w.jpg" else "p_abpafnpyfUaYs4kCSF7w_mean.npy"
            )
            logits_mm = np.load(mean_file, mmap_mode='r')   # ≈ 0 MB resident
            domain_pred = np.argmax(logits_mm, axis=0)      # np handles mmap
            del logits_mm                                    # release file handle
        else:
            domain_pred = np.array(
                Image.open(os.path.join(
                    "../data/mapillary_v1/validation/labels/",
                    file_name.replace(".jpg", ".png")
                ))
            ).astype(np.int64)

        logger.info(f"[Domain loaded]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

        covered_merge_tensor, full_merge_tensor, local_uncertain_percentile_np, _ = core.domain_vote_per_region(
            overlayed_mask_np=np.load(overlayed_mask_dir),
            sam_masks=None, 
            domain_pred=torch.from_numpy(domain_pred).to(device), 
            uncertainty_np=np.load(
                os.path.join(
                    "../results/",
                    f"{args.domain_name}" if args.domain_name != "ground-truth" else "mask2former",
                    f"{args.data_grp}",
                    f"{file_name.replace('.jpg', '_mut.npy')}" if file_name != "p_abpafjpgfUaYs4kCSF7w.jpg" else "p_abpafnpyfUaYs4kCSF7w_mut.npy"
                )), 
            uncertainty_strategy=args.uncertainty_strategy,
            author=args.sam_author,
            rank=rank
        )
        logger.info(f"[Merged]: {file_name}; {i} / {len(local_img)}; on rank {rank} / {args.world_size}")

        del local_uncertain_percentile_np
        
        for j, item in enumerate([covered_merge_tensor, full_merge_tensor]):
            visuals.draw_save_prd(
                segm_binary_mask=item,
                save_mask_dir=os.path.join("../results", root_dir, args.data_grp, 'npy', 'labl', file_name.replace('.jpg', f'_{j}.npy')),
                grid_points=None,
                img_array=mmcv.imread(file_path, backend='pillow'), # img_array
                save_img_dir=os.path.join("../results", root_dir, args.data_grp, 'img', file_name.replace('.jpg', f'_{j}.jpg')),
                rank=rank,
                granular='low'
            )
            logger.info(f"[Success]: {file_name}; {i}, {j} / {len(local_img)}; on rank {rank} / {args.world_size}")

        # ---------------------------------------------------------
        # Explicit memory cleanup after finishing this image
        # ---------------------------------------------------------
        try:
            del overlayed_mask_np, domain_pred, covered_merge_tensor, full_merge_tensor
        except NameError:
            pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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
        mp.spawn(main, args=(root_dir, args,), nprocs=args.world_size,join=True)
    else:
        main(0, root_dir, args)

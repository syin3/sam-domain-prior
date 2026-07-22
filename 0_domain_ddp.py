"""
Forward pass of domain model; saves predicted probability guaranteed; extract max_class or save predicted clas depend on user input.

Usage:
(1) python 0_domain_ddp.py --data_dir "../data/mapillary_v1" --world_size 8 --data_grp validation --num_img 2000 --model_name mask2former --mc True --mc_num 5 --granular high --max_class False
(2) python 0_domain_ddp.py --data_dir "../data/mapillary_exm" --world_size 1 --data_grp validation --num_img 10 --model_name deeplab --mc True --mc_num 10 --granular high --max_class False
"""
import sys, os, argparse, time
import numpy as np
import torch
from pathlib import Path

from PIL import Image
import mmcv

from src import utils, core, metrics
from src.configs import FB_CONFIG, MMLAB_CONFIG

import torch.distributed as dist
import torch.multiprocessing as mp

import logging
logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("domain_ddp.log", mode="w"),
    ],
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Domain inference')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--data_grp', type=str, default='both', choices=['training', 'validation'], help='group of data to inference')
    parser.add_argument('--num_img', type=int, default=0, help='number of images to process by all nodes')
    parser.add_argument('--model_name', type=str, default='mask2former', choices=['mask2former', 'deeplab'], help='which domain seg model to load')
    parser.add_argument('--mc', type=str, default='True', choices=['True', 'False'], help='whether to turn on the MC mode of the domain model')
    parser.add_argument('--mc_num', type=int, default=1, help='number of passes for MC estimation (if 1)')
    parser.add_argument('--granular', type=str, default='high', choices=['high', 'low'], help='high - int64 and float32; low - np.unit8 and float 16')
    parser.add_argument('--max_class', type=str, default='True', choices=['True', 'False'], help='save prob and index of the max class only or not')
    args = parser.parse_args()
    return args

def main(rank, args):
    """
    size:
        prd.size() - [65, 3000, 4000]
        prd.argmax(dim=0).size() - [3000, 4000]
        prd.argmax(dim=0, keepdim=True).size() - [1, 3000, 4000]
    """
    assert args.mc_num > 1 if args.mc == "True" else args.mc_num > 0
    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    train_files = [fn_ for fn_ in os.listdir(os.path.join(args.data_dir, args.data_grp, 'images')) if '.jpg' in fn_]
    if args.num_img > 0:
        train_files = train_files[:args.num_img]

    local_files = train_files[
        (len(train_files) // args.world_size) * rank : (len(train_files) // args.world_size) * (rank + 1)
    ]
    logger.info(f"[Img] {len(local_files):,} image files identified for device {rank}")

    artifact = core.load_domain(name=args.model_name, rank=rank)
    model = artifact['model']
    model.eval()

    if args.mc == 'True': # turn on dropout and droppath during inference
        if args.model_name == 'deeplab': # deeplab
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
                    module.train()
        else: # mask2former
            from transformers.models.swin.modeling_swin import SwinDropPath
            for module in model.modules():
                if isinstance(module, SwinDropPath):
                    module.train()
        logger.info(f"[Simu] MC status turned on for device: {rank}")

    for i, file_name in enumerate(local_files):
        # models can also request image read in by certain packages
        # img_pil = Image.open(file_path);
        img_path = os.path.join(args.data_dir, args.data_grp, 'images', file_name)

        with torch.no_grad():
            for j in range(args.mc_num): # number of mc run

                if os.path.exists(
                    os.path.join("../results/", f"{args.model_name}", f"{args.data_grp}", str(j), "npy", "prob", file_name.replace('jpg', 'npy'))
                ):
                    logger.info(f"[Skip]: {file_name}; {i} / {len(local_files)}; run {j}; on rank {rank} / {args.world_size}; prob exists")
                    continue
                
                try:
                    logits = core.infer_domain(
                        img_input=img_path,
                        model_name=args.model_name,
                        artifact=artifact,
                        rank=rank)
                except Exception as e:
                    logger.error(f"[Error]: {file_name}; {i} / {len(local_files)}; run {j}; on rank {rank} / {args.world_size}; {e}")
                    continue

                # logits --> prob
                prd = logits.softmax(dim=0) # cuda:0
                
                if args.max_class == "True": # if only save prob and indices for the max class
                    prd = prd.max(dim=0)
                    prd_prob = prd.values.cpu().numpy() # torch.float32 --> float16 should be okay
                    prd_class = prd.indices.cpu().numpy() # torch.int64 --> np.unit8 shoul be okay for evaluation
                else:
                    prd_prob = prd.cpu().numpy()
                    prd_class = None

                # save predicted probabilities
                np.save(
                    os.path.join("../results/", f"{args.model_name}", f"{args.data_grp}", str(j), "npy", "prob", file_name.replace('jpg', 'npy')),
                    prd_prob if args.granular == 'high' else prd_prob.astype(np.float16)
                )

                # save predicted classes
                if prd_class is not None:
                    core.draw_save_prd( # save the predicted classes and draw to image
                        mask=prd_class,
                        mask_dir=os.path.join("../results/", f"{args.model_name}/", args.data_grp, str(j), 'npy', 'labl', file_name.replace('jpg', 'npy')),
                        img_array=mmcv.imread(img_path, backend='pillow'),
                        save_img_dir=os.path.join("../results/", f"{args.model_name}/", args.data_grp, str(j), 'img', file_name),
                        rank=rank,
                        granular=args.granular
                    )

        logger.info(f"[Success]: {file_name}; {i} / {len(local_files)}; on rank {rank} / {args.world_size}")


if __name__ == '__main__':
    args = parse_args()

    for j in range(args.mc_num):
        Path("../results/", f"{args.model_name}/", args.data_grp, str(j), 'img').mkdir(parents=True, exist_ok=True)
        Path("../results/", f"{args.model_name}/", args.data_grp, str(j), 'npy', 'labl').mkdir(parents=True, exist_ok=True)
        Path("../results/", f"{args.model_name}/", args.data_grp, str(j), 'npy', 'prob').mkdir(parents=True, exist_ok=True)
        logger.info("[Path] Folder directories created")

    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)

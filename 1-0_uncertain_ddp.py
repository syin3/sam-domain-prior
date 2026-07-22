"""
Aggregate results of several forward passes (npy files) into mean, var, predictive entropy, and mutual infomation.

Usage:
(1) python 1-0_uncertain_ddp.py --world_size 8 --data_grp validation --start_img_idx 1600 --end_img_idx 2000 --model_name mask2former --mc_num 5
"""
import sys, os, argparse, time
import numpy as np
import torch
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

import logging
logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("uncertain_ddp.log", mode="w"),
    ],
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Uncertainty quantification')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--data_grp', type=str, default='both', choices=['training', 'validation'], help='group of data to inference')
    parser.add_argument('--start_img_idx', type=int, default=0, help='start index of images to process by all nodes')
    parser.add_argument('--end_img_idx', type=int, default=0, help='end index of images to process by all nodes')
    parser.add_argument('--model_name', type=str, default='mask2former', help='which domain seg model to load')
    parser.add_argument('--mc_num', type=int, default=1, help='number of passes for MC estimation (if 1)')
    args = parser.parse_args()
    return args

def main(rank, args):
    """
    each GPU is responsible for processing a subset of images
    NOT each GPU processes a MC pass of the full set
    """
    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # forge base directory
    base_img_dir = os.path.join(
        "../results",
        args.model_name,
        args.data_grp,
        "{}",
        "npy",
        "prob"
    )

    # output_dir and suffixes
    output_dir = Path("../results", args.model_name)
    suffixes = ['_mean', '_mut'] # '_var', '_ent',

    # decide the files
    untouched_files = []
    train_files = [fn_ for fn_ in os.listdir(base_img_dir.format(0)) if '.npy' in fn_]

    train_files = train_files[args.start_img_idx:args.end_img_idx]
    for file_name in train_files:
        if all(
            (output_dir / args.data_grp / f"{file_name.replace('.npy', '')}{suffix}.npy").exists()
            for suffix in suffixes
        ):
            continue
        else:
            untouched_files.append(file_name)

    # split the images for each GPU
    local_files = untouched_files[
        (len(untouched_files) // args.world_size) * rank : (len(untouched_files) // args.world_size) * (rank + 1)
    ]
    logger.info(f"[Img] {len(local_files):,} image files identified for device {rank}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # process each identified image
    # due to CUDA OOMs, stream basic stats and produce mean, variance, sample_entropy at the end
    for i, file_name in enumerate(local_files):
        # Check if all output files already exist
        base_name = file_name.replace('.npy', '')
            
        with torch.no_grad():
            for j in range(args.mc_num):
                npy_path = os.path.join(base_img_dir.format(j), file_name)
                # log error if the file does not exist
                if not os.path.exists(npy_path):
                    logger.error(f"[File] MC {j} of {file_name} does not exist")
                    continue
                p = torch.from_numpy(np.load(npy_path))# .half()  # Half precision
                p = p.to(device, non_blocking=True)

                # Memory-efficient entropy calculation
                with torch.cuda.amp.autocast():  # Mixed precision
                    e = torch.special.entr(p).sum(dim=0)  # More memory-efficient than manual calculation

                if p.ndim < 3:
                    logger.error(f"[Dimension] tensor of MC {j} for {file_name} only has ndim {p.ndim} < 3")
                    continue

                if j == 0:
                    p_sum = p.float()
                    p_squared_sum = p.square().float()
                    e_sum = e.float()
                else:
                    p_sum += p
                    p_squared_sum += p.square()
                    e_sum += e

                del p,e
                torch.cuda.empty_cache()

            p_mean = p_sum.cpu() / args.mc_num
            p_variance = 1/(args.mc_num-1) * (p_squared_sum.cpu() - args.mc_num * p_mean.square().cpu()) # https://pytorch.org/docs/stable/generated/torch.var.html
            e_mean = e_sum.cpu() / args.mc_num # average sample entropy
            p_entropy = -(p_mean * p_mean.log()).sum(dim=0)
            p_muinfo = p_entropy - e_mean
            assert p_mean.ndim == 3 # mean
            assert p_variance.ndim == 3 # variance
            assert p_entropy.ndim == 2 # predictive entropy
            assert p_muinfo.ndim == 2 # mutual infomation

            for tensor, suffix in zip([p_mean, p_variance, p_entropy, p_muinfo],
                                    suffixes):
                save_path = output_dir / args.data_grp / f"{base_name}{suffix}.npy"
                np.save(str(save_path),tensor.numpy())
            
            logger.info(f"[Success]: {file_name}; {i} / {len(local_files)}; on rank {rank} / {args.world_size}")

if __name__ == '__main__':
    args = parse_args()

    if args.world_size > 1:
        mp.spawn(main,args=(args,), nprocs=args.world_size,join=True)
    else:
        main(0, args)


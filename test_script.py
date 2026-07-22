import os
import time

import numpy as np
from tabulate import tabulate

from src import core

pred_list, gt_list = [], []

base_dir_gt = "../data/mapillary_v1/validation/labels/"

def print_metrics_one(metrics):
    # Overall metrics
    print("\nOVERALL METRICS:")
    print(tabulate([(k, v) for k, v in metrics.items() if not isinstance(v, dict)],
                   headers=["Metric", "Value"], floatfmt=".3f"))

    # Class metrics
    class_data = []
    for cls in [k for k in metrics if isinstance(metrics[k], dict)]:
        row = [cls] + [metrics[cls].get(h, 'N/A') for h in ['IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']]
        class_data.append(row)

    print("\nCLASS-WISE METRICS:")
    print(tabulate(class_data,
                 headers=["Class", "IoU", "Acc", "Dice", "Fscore", "Precision", "Recall"],
                 floatfmt=".3f"))


def print_metrics_multiple(model_metrics_list, model_names):
    """
    model_metrics_list: List of metrics dictionaries for each model
    model_names: List of model names (e.g., ["SAM", "DeepLab", "Mask2Former"])
    """

    # Build consolidated rows
    table_data = []

    # 1. Add overall metrics first
    overall_metrics = [
        'aAcc', 'mAcc', 'mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore',
        'aAcc_select', 'mIoU_select', 'mDice_select', 'mPrecision_select', 'mRecall_select', 'mFscore_select'
    ]
    for metric in overall_metrics:
        row = [metric] + [m.get(metric, np.nan) for m in model_metrics_list]
        table_data.append(row)

    # 2. Add class-specific metrics
    classes = [k for k in model_metrics_list[0] if isinstance(model_metrics_list[0][k], dict)]

    for cls in classes:
        # Add class name as header row
        table_data.append([f"→ {cls}", *[""]*(len(model_names))])

        # Add each metric for the class
        for metric in ['IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']:
            row = [f"  {metric}"]  # Indent metric names
            for model_metrics in model_metrics_list:
                value = model_metrics[cls].get(metric, np.nan)
                row.append(round(value, 3) if not np.isnan(value) else "N/A")
            table_data.append(row)

    # Create header with model names
    headers = ["Metric/Class"] + model_names

    # Print the table
    print(tabulate(table_data,
                 headers=headers,
                 tablefmt="grid",
                 floatfmt=".3f",
                 stralign="left",
                 missingval="N/A"))


def calc_metrics_one(base_dir):
    """
    full validation set
        * deeplabv3+ official: {"aAcc": 90.8, "mIoU": 47.35, "mAcc": 56.21, "step": 300000}
        * deeplabv3+ ours: 90.8, 44.8, 56.2
        * mask2former ours: 91.4, 48.8, 64.6
    """
    start = time.time()
    files = [fn_ for fn_ in os.listdir(base_dir) if '.npy' in fn_]

    res = \
        core.eval_mask_list(
        [os.path.join(base_dir, item) for item in files],
        [os.path.join(base_dir_gt, item.replace('npy', 'png')) for item in files],
        include_index_list=[2,3,4,5,6,7,8,10,12,13,14,15,23,24,36,38,39,41,43,44,45,46,47,48,50]
    )
    print(f"Evaluation time: {time.time() - start:.1f} s")
    return res

def main():
    # baseline: Mask2former, Deeplabv3+
    print_metrics_one(
        # calc_metrics_one("/Volumes/T7/dissertation/results/domain_mask2former_2k/validation/npy/labl")
        calc_metrics_one("/Volumes/T7/dissertation/results/domain_deeplab_2k/validation/npy/labl")
    )
    # models = [
    #     "/Volumes/T7/dissertation/domain_mask2former_2k/validation/npy/labl",
    #     "/Volumes/T7/dissertation/domain_deeplab_2k/validation/npy/labl",
    #     # "../results/post-pre/mask2former_relative_0.2-0.5_b_128_0_1.0/validation/npy/labl",
    #     # "../results/post-pre/mask2former_relative_0.2-0.5_h_128_0_1.0/validation/npy/labl",
    #     # "../results/post-pre/mask2former_all_h_128_0_1.0/validation/npy/labl",
    # ]
    # print_metrics_multiple(
    #     [calc_metrics_one(item) for item in models],
    #     [item.split("/")[5] for item in models]
    # )

    # 0_1.0 and 1_1.2 are the same
    # print_metrics_one(
    #     calc_metrics_one("/Volumes/T7/dissertation/results/ground-truth_all_l_128_0_1.0/validation/npy/labl")
    #     # calc_metrics_one("/Volumes/T7/dissertation/results/ground-truth_all_l_128_1_1.2/validation/npy/labl")
    # )

    # 3.3.1 ground truth upper limit
    # models = [
    #     "/Volumes/T7/dissertation/results/ground-truth_all_b_128_0_1.0/validation/npy/labl",
    #     "/Volumes/T7/dissertation/results/ground-truth_all_b_64_0_1.0/validation/npy/labl",
    #     "/Volumes/T7/dissertation/results/ground-truth_all_b_32_0_1.0/validation/npy/labl",
    #     "/Volumes/T7/dissertation/results/ground-truth_all_b_16_0_1.0/validation/npy/labl",
    # ]
    # print_metrics_multiple(
    #     [calc_metrics_one(item) for item in models],
    #     [item.split("/")[5] for item in models]
    # )

if __name__ == '__main__':
    main()
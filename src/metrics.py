import os
from collections import OrderedDict

import numpy as np
import torch

import mmcv

from src import visuals

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_uncertain_metrics(
    img,
    base_dir="~/Desktop/mask2former/validation/",
    use_mean_argmax=True,
    top_k = 3
):
    """Calculate uncertainty metrics.

    Usage:
        metrics.uncertain_metrics("RDClvieQTUVfSvdrteZmPA", "../results/deeplab/validation")

    Return:
        chosen class (defaul to argmx of mean probability)
        var
            of chosen class
            sum of top-k classes
            sum of all class
        predictive entropy
        mutual information
    """

    mean_probs = np.load(os.path.join(base_dir, f"{img}_mean.npy"))

    if use_mean_argmax:
        chosen_class = np.argmax(mean_probs, axis=0)
    else:
        chosen_class = get_class_by_mode(
            img,
            base_dir=base_dir + "{}/npy/prob/",
            num_classes=66
        )

    # variance
    var_of_chosen_class = np.take_along_axis(
        np.load(os.path.join(base_dir, f"{img}_var.npy")),
        chosen_class[None, ...],
        axis=0
    ).squeeze(0)

    top_k_indices = np.argsort(mean_probs, axis=0)[-top_k:]
    # Sum variances across top K classes
    var_sum_of_top_k_class = np.take_along_axis(
        np.load(os.path.join(base_dir, f"{img}_var.npy")),
        top_k_indices,
        axis=0
    ).sum(axis=0)

    # sum variances across all classes
    var_sum_of_all_classes = np.sum(np.load(os.path.join(base_dir, f"{img}_var.npy")), axis=0)

    # predictive entropy
    total_entropy = np.load(os.path.join(base_dir, f"{img}_ent.npy"))

    # mutual information
    mutual_information = np.load(os.path.join(base_dir, f"{img}_mut.npy"))

    return chosen_class, \
        var_of_chosen_class, var_sum_of_all_classes, var_sum_of_top_k_class, \
        total_entropy, mutual_information

def get_class_by_mode(
    img,
    base_dir="~/Desktop/mask2former/validation/{}/npy/prob/",
    mc_num=10,
    num_classes=66
):
    """Compute mode across 10 iterations with reduced memory footprint."""
    # Get array dimensions from first file
    sample = np.argmax(
        np.load(os.path.join(base_dir.format(0), f"{img}.npy"),
        axis=0)
    )
    H, W = sample.shape

    # Initialize count matrix (height, width, class)
    counts = np.zeros((H, W, num_classes), dtype=np.uint8)
    np.add.at(counts, (np.arange(H)[:, None], np.arange(W), sample), 1)

    # Incrementally update counts
    for i in range(1, mc_num):
        arr = np.argmax(np.load(os.path.join(base_dir.format(i), f"{img}.npy"), axis=0))
        # Vectorized counting using numpy advanced indexing
        np.add.at(counts, (np.arange(H)[:, None], np.arange(W), arr), 1)

    # Find mode indices (class with max count)
    mode_result = counts.argmax(axis=-1)
    return mode_result.astype(sample.dtype)

# def f_score(precision, recall, beta=1):
#     """calculate the f-score value.

#     Args:
#         precision (float | torch.Tensor): The precision value.
#         recall (float | torch.Tensor): The recall value.
#         beta (int): Determines the weight of recall in the combined score.
#             Default: False.

#     Returns:
#         [torch.tensor]: The f-score value.
#     """
#     score = (1 + beta**2) * (precision * recall) / (
#         (beta**2 * precision) + recall)
#     return score

def f_score(precision, recall, beta=1):
    """Calculate F-score with proper handling of zero divisions."""
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    
    # Handle cases where both precision and recall are zero
    zero_mask = (precision == 0) & (recall == 0)
    score = torch.zeros_like(denominator)
    valid_mask = ~zero_mask
    
    # Only calculate for valid entries
    score[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return score


def intersect_and_union_per_file(
    pred_label,
    label,
    num_classes,
    ignore_index,
    label_map=dict(),
    reduce_zero_label=False
):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation. -> list
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label)).to(DEVICE)
    else:
        pred_label = torch.from_numpy((pred_label)).to(DEVICE)

    if isinstance(label, str):
        label = torch.from_numpy(mmcv.imread(label, flag='unchanged', backend='pillow')).to(DEVICE)
    else:
        label = torch.from_numpy(label).to(DEVICE)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]

    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              # ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation. -> list
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64).to(DEVICE)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64).to(DEVICE)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64).to(DEVICE)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64).to(DEVICE)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union_per_file(
                result, gt_seg_map, num_classes, ignore_index=66, # always ignore unlabeled
                label_map=label_map, reduce_zero_label=reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          include_index_list=[0],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # generate additional aggregate metrics for selected indices
    total_area_intersect_select = total_area_intersect[include_index_list]
    total_area_union_select = total_area_union[include_index_list]
    total_area_pred_label_select = total_area_pred_label[include_index_list]
    total_area_label_select = total_area_label[include_index_list]

    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    dice = (2 * total_area_intersect) / (total_area_pred_label + total_area_label)
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    fscore = f_score(precision, recall, beta)

    # For selected indices
    acc_select = total_area_intersect_select / total_area_label_select
    iou_select = total_area_intersect_select / total_area_union_select
    dice_select = (2 * total_area_intersect_select) / (total_area_pred_label_select + total_area_label_select)
    precision_select = total_area_intersect_select / total_area_pred_label_select
    recall_select = total_area_intersect_select / total_area_label_select
    fscore_select = f_score(precision_select, recall_select, beta)

    # average/mean metrics
    ret_metrics = OrderedDict(
        {'aAcc': (total_area_intersect.sum() / total_area_label.sum()).item(),
         'mAcc': np.nanmean(acc),
         'mIoU': np.nanmean(iou),
         'mDice': np.nanmean(dice),
         'mPrecision': np.nanmean(precision),
         'mRecall': np.nanmean(recall),
         'mFscore': np.nanmean(fscore),
         'aAcc_select': (total_area_intersect_select.sum() / total_area_label_select.sum()).item(),
         'mIoU_select': np.nanmean(iou_select),
         'mDice_select': np.nanmean(dice_select),
         'mPrecision_select': np.nanmean(precision_select),
         'mRecall_select': np.nanmean(recall_select),
         'mFscore_select': np.nanmean(fscore_select),
        }
    )

    # per-class metrics
    ret_metrics["Acc"] = acc.cpu().numpy()
    ret_metrics["IoU"] = iou.cpu().numpy()
    ret_metrics["Dice"] = dice.cpu().numpy()
    ret_metrics["Fscore"] = fscore.cpu().numpy()
    ret_metrics["Precision"] = precision.cpu().numpy()
    ret_metrics["Recall"] = recall.cpu().numpy()

    # convert nan to num if instruction given
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    
    return ret_metrics

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 include_index_list,
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation. -> list
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = \
        total_intersect_and_union(
            results, gt_seg_maps, num_classes,
            # ignore_index, # no need to pass ignore_index down, we're post-processing later
            label_map, reduce_zero_label
        )

    # we need to exclude a list rather than just one class
    # one way is to modify intersect_and_union() and only compute for wanted classes, ~np.isin(label, ignore_index_list)
    # the other is to have every class computed and post-process them
    # see the _select metrics in metrics.total_area_to_metrics()
    ret_metrics = total_area_to_metrics(total_area_intersect,
                                        total_area_union,
                                        total_area_pred_label,
                                        total_area_label,
                                        include_index_list,
                                        nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics

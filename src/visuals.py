import os
from pathlib import Path
import torch, mmcv
from PIL import Image
import numpy as np

import cv2
import matplotlib
import matplotlib.pyplot as plt

from src import utils

import logging
logger = logging.getLogger(__name__)

MMSEG_DIR = "../mmsegmentation"
CKPTS_DIR = "../ckpts"


def draw_save_prd(
    segm_binary_mask: np.array, 
    save_mask_dir: str, 
    grid_points: np.array,
    img_array: np.array, 
    save_img_dir: str, 
    rank: int, 
    granular: str, 
    show_class_name: bool=True):
    """save predicted class and image to designated locations"""

    # save np
    if save_mask_dir is not None:
        np.save(save_mask_dir, segm_binary_mask.cpu().numpy() if granular == 'high' else segm_binary_mask.cpu().numpy().astype(np.uint8))
        logger.info("[Save] Predicted classes saved")

    # save img
    if save_img_dir is not None:
        x, y, z = utils.get_final_mask(segm_binary_mask, rank)
        img = utils.imshow_det_bboxes(
            img_array,
            bboxes=None,
            labels=x, # np.arange(len(semantic_cls_in_img)),
            segms=np.stack(y),
            grid_points=grid_points,
            grid_size=400,
            grid_color='red',
            class_names=z if show_class_name else None,
            mask_color='mapillary',
            font_size=15,
            show=False,
            out_file=save_img_dir)

        logger.info("[Save] Predicted classes drawn to image")
    
    return img

def draw_heatmap(mask, img_array, save_img_dir=None, alpha=0.8):
    """Given a tensor of metric, draw on image to show heat map.

    Refer to: https://github.com/xmed-lab/CLIP_Surgery/blob/master/demo.ipynb
    
    mask : torch.tensor, (H,W), values are either [0,1] or [0,255]
    img_array : np.ndarray, (H, W, C), unit8
    save_img_dir : str, path to save image
    rank : int, GPU id
    """
    # check if mask is torch.tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy() if mask.device.type == 'cpu' else mask.cpu().numpy()
    
    # normalize to [0,255]
    mask_plot = mask / mask.max() * 255
    mask_plot = mask_plot.astype('uint8')

    color_mask = cv2.applyColorMap(mask_plot, cv2.COLORMAP_JET)
    bgr = img_array * (1 - alpha) + color_mask * alpha
    bgr = bgr.astype('uint8')
    
    # no need to convert, cv2.imwrite() expects BGR
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) 

    # Save using OpenCV's imwrite
    if save_img_dir is not None:
        if not os.path.exists(save_img_dir):
            Path(save_img_dir).parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(save_img_dir, bgr)
        logger.info(f"[Save] Heatmap saved to drawn to image")
    


def two_pane_heatmap(
    heatmap_dict, 
    use_cbar=True,
    orig_img_array=None, overlap_flag=False, overlap_alpha=0.8
):
    """Visualize one or two heatmaps with optional original image overlay.
    
    Args:
        heatmap_dict: Dict, [str, 2D np array]
    """
    num_plots = len(heatmap_dict)
    if num_plots == 0:
        return

    # Create dynamic subplot layout
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    axes = [axes] if num_plots == 1 else axes

    # Configure scientific notation formatter
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))

    for ax, (name, value) in zip(axes, heatmap_dict.items()):
        # Show original image underlay if requested
        if overlap_flag and orig_img_array is not None:
            ax.imshow(mmcv.bgr2rgb(orig_img_array), alpha=overlap_alpha)
            
        # Plot heatmap
        im = ax.imshow(value, cmap='viridis' if name == 'Entropy' else 'jet', alpha=0.9 if overlap_flag else 1.0)
        ax.set_title(name)
        ax.axis('off')
        
        # Add formatted colorbar
        if use_cbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.5, format=formatter)
            cbar.ax.tick_params(labelsize=9)
            cbar.update_ticks()

    plt.tight_layout()
    plt.show()

def plot_uncertainty_analysis(uncertain_pred, class_pred, class_gt, xlabel, is_split: bool = False):
    # Common calculations
    logbins = np.logspace(-6, -2, 20)
    total_counts, _ = np.histogram(uncertain_pred.flatten(), bins=logbins)
    misclass_counts, _ = np.histogram(uncertain_pred[class_pred != class_gt], bins=logbins)
    misclass_rate = misclass_counts / (total_counts + 1e-8)
    bin_centers = np.sqrt(logbins[:-1] * logbins[1:])

    if is_split:
        # Original separate plots
        fig1, ax3 = plt.subplots()
        ax3.hist(uncertain_pred.flatten(), bins=logbins, alpha=0.5, color='orange', label='all')
        ax3.hist(uncertain_pred[class_pred != class_gt], bins=logbins, alpha=0.5, color='blue', label='misclassified')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.set_ylabel('Num of pixels per Bin')
        ax3.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.title('Uncertainty Distribution')
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(bin_centers, total_counts, 's--', color='orange', label='Total Samples')
        ax1.set_ylabel('Num of pixels per Bin', color='orange')
        ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        ax1.tick_params(axis='y', labelcolor='orange')

        # Create a twin axis for total counts
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, misclass_rate * 100, 'o-', color='blue', label='Misclassification Rate')
        ax2.set_xscale('log')
        ax2.set_ylabel('Misclassification Rate (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Add combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + lines2, labels + labels2, loc='lower center', 
            bbox_to_anchor=(0.5, 0.01),  # 1% from bottom
            #ncol=2
        )

        ax1.set_xlabel(xlabel)
        plt.title('Misclassification Rate vs Sample Distribution')
        plt.show()
        
    else:
        # Combined subplots
        fig, (ax_hist, ax_analysis) = plt.subplots(1, 2, figsize=(13, 6))
        
        # Top: Histogram plot
        ax_hist.hist(uncertain_pred.flatten(), bins=logbins, alpha=0.5, color='orange', label='all')
        ax_hist.hist(uncertain_pred[class_pred != class_gt], bins=logbins, alpha=0.5, color='blue', label='misclassified')
        ax_hist.set_xscale('log')
        ax_hist.set_xlabel(xlabel)
        ax_hist.legend()
        ax_hist.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        # ax_hist.set_title('Uncertainty Distribution')

        # Bottom: Analysis plot with twin axis
        ax1 = ax_analysis
        ax1.plot(bin_centers, total_counts, 's--', color='orange', label='Total Samples')
        ax1.set_ylabel('Total Samples', color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')
        ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, misclass_rate * 100, 'o-', color='blue', label='Misclassification Rate')
        ax2.set_ylabel('Misclassification Rate (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + lines2, labels + labels2, 
            loc='best', 
            # bbox_to_anchor=(0.5, 0.01)
        )
        ax1.set_xlabel(xlabel)
        # ax1.set_title('Misclassification Analysis')
        ax1.set_xscale('log')
        
        plt.tight_layout()
        plt.show()

def two_sided_log_bins(min_val, max_val, n_neg=10, n_pos=10, min_magnitude=1e-12):
    """
    Create bin edges using a log scale for negative values (by magnitude) and a
    log scale for positive values. Insert 0 as a boundary in between.

    Parameters
    ----------
    min_val : float
        The minimum negative value (e.g. -2e-6). Must be < 0.
    max_val : float
        The maximum positive value (e.g. 1e-2). Must be > 0.
    n_neg : int
        Number of log-spaced bins on the negative side.
    n_pos : int
        Number of log-spaced bins on the positive side.
    min_magnitude : float
        A small positive magnitude for the innermost log bin edge near zero.
        By default, 1e-12 to be safely below typical tiny values.

    Returns
    -------
    bins : np.ndarray
        A 1D array of bin edges that goes from min_val (negative) up to 0,
        then from 0 up to max_val (positive), both sides in log spacing.
    """
    if min_val >= 0:
        raise ValueError("min_val must be negative.")
    if max_val <= 0:
        raise ValueError("max_val must be positive.")

    # 1) Negative side: use log of the magnitude (from min_magnitude to abs(min_val))
    mag_neg_min = max(min_magnitude, 1e-16)  # avoid log10(0)
    mag_neg_max = abs(min_val)
    # Create log-spaced magnitudes on [mag_neg_min, mag_neg_max]
    magnitudes_neg = np.logspace(np.log10(mag_neg_min), np.log10(mag_neg_max), n_neg)

    # Convert magnitudes to negative, then reverse order
    # so the array goes from min_val (largest magnitude, negative) up to near zero.
    bins_neg = -magnitudes_neg[::-1]

    # 2) Positive side: from min_magnitude to max_val
    mag_pos_min = max(min_magnitude, 1e-16)
    mag_pos_max = max_val
    magnitudes_pos = np.logspace(np.log10(mag_pos_min), np.log10(mag_pos_max), n_pos)

    # 3) Combine negative bins, zero, and positive bins
    bins = np.concatenate([bins_neg, [0], magnitudes_pos])
    
    # Ensure strictly increasing bin edges and remove any potential duplicates
    bins = np.unique(bins)
    return bins

def two_sided_log_bin_centers(bin_edges):
    """
    Given bin edges that include negative, zero, and positive values in a 
    two-sided log fashion, compute a reasonable 'center' for each bin.

    Returns
    -------
    centers : np.ndarray
        The center for each bin, where bins[i] <= center[i] < bins[i+1].
    """
    centers = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        # Case 1: Entirely negative bin
        if right < 0:
            # Use negative of geometric mean of the absolute values
            c = -np.sqrt(abs(left) * abs(right))

        # Case 2: Entirely positive bin
        elif left > 0:
            # Use geometric mean
            c = np.sqrt(left * right)

        else:
            # Case 3: The bin crosses 0 or touches 0
            # e.g. [-1e-7, 0], or [0, 1e-9], or [-1e-9, 1e-9]
            # A simple approach: the midpoint
            c = 0.5 * (left + right)

            # Alternatively, you can just pick 0 if the interval crosses zero
            # c = 0

        centers.append(c)

    return np.array(centers)
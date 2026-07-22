import os
from pathlib import Path
import torch, mmcv
from PIL import Image
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
from typing import Optional, List, Any, Tuple

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline
import pycocotools.mask as mask_utils

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model

import cv2
import matplotlib
import matplotlib.pyplot as plt

from src.configs import FB_CONFIG, MMLAB_CONFIG
from src import utils, metrics

import logging
logger = logging.getLogger(__name__)

MMSEG_DIR = "../mmsegmentation"
CKPTS_DIR = "./segment-anything/ckpts"

from concurrent.futures import ThreadPoolExecutor
import math

def load_sam(author, size, rank, **kwargs):
    """wrapper to load sam model"""
    if author == 'facebook':
        return load_sam_fb(
            name=size,
            pts_per_side=kwargs.get("pts_per_side", 64),
            pts_per_batch=kwargs.get("pts_per_batch", 64),
            crop_n_layers=kwargs.get("crop_n_layers", 0),
            crop_n_points_downscale_factor=kwargs.get("crop_n_points_downscale_factor", 1.2),
            rank=rank
        )
    else:
        return load_sam_hf(
            name=size,
            rank=rank
        )

def load_sam_fb(name, **kwargs):
    """
    load FB implementation of SAM

    pts_per_side determines the sampling density -> quality as well as computation load
    """
    if name == 'vit_b':
        model_name = 'sam_vit_b_01ec64'
    elif name == 'vit_l':
        model_name = 'sam_vit_l_0b3195'
    else:
        model_name = 'sam_vit_h_4b8939'

    device = torch.device(f"""cuda:{kwargs.get("rank", 0)}""" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[name](checkpoint=os.path.join(CKPTS_DIR, f"{model_name}.pth")).to(device)

    artifact = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=kwargs["pts_per_side"],
        points_per_batch=kwargs["pts_per_batch"],
        pred_iou_thresh=kwargs["pred_iou_thresh"],
        stability_score_thresh=kwargs["stability_score_thresh"],
        crop_n_layers=kwargs["crop_n_layers"],
        crop_n_points_downscale_factor=kwargs["crop_n_points_downscale_factor"],
        min_mask_region_area=kwargs["min_mask_region_area"],
        output_mode='coco_rle', # TODO: change to default binary mask
    )

    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        assert str(next(artifact.predictor.model.parameters()).device) == "cuda:0"

    logger.info(f"[Loaded] FB - {model_name} to {sam.device}")
    return artifact

def load_sam_hf(name, rank=0):
    """
    load HF implementation of SAM

    hyperparameters are not specified now because HF implementation only involves them during function call
    """
    sam_full_name_dict = {'h':'huge', 'l':'large', 'b':'base'}
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    artifact = pipeline("mask-generation", model=f"facebook/sam-vit-{sam_full_name_dict[name[-1]]}", device=device)
    logger.info(f"[Loaded] HF - {name} to {device}")

    return artifact

def infer_sam(file_path, author, model, **kwargs):
    """
    hugging face's implementation has two issues
    (1) crops_n_layers vs. crop_n_layers: naming of this parameter is inconsistent across functions
    (2) crop_n_layers > 0 & crop_n_points_downscale_factor != 1.0 will throw dimension error because 
        it requires point grids from all crops have consistent dimensions "points_per_crop = np.array([point_grid_per_crop])",
        which is certainly not the case if "crop_n_points_downscale_factor != 1.0".
        This isn't an issue in FB's implementation because it processes per item in the list.

    References
    (1) https://huggingface.co/docs/transformers/main/en/model_doc/sam
    (2) https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
    (3) https://huggingface.co/docs/transformers/en/tasks/mask_generation
    """
    # Ensure filter_mask is properly handled (may be None)
    filter_mask = kwargs.get("filter_mask", None)
    
    if author == 'facebook':
        # artifact = core.load_sam_fb("vit_b", crop_n_layers=0, crop_n_points_downscale_factor=1.2)
        # core.infer_sam('../data/mapillary_exm/validation/images/dejslqZtqGZ53SiXTInqug.jpg', author="facebook", model=artifact)
        masks = model.generate(
            mmcv.imread(file_path, backend='pillow'),
            filter_mask=filter_mask
        )
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        return {
            "masks": [item for item in sorted_masks],
            "grid_points": [item["point_coords"] for item in sorted_masks]
        }
    else: # hugging-face
        # artifact = core.load_sam_hf("bit_b")
        # core.infer_sam('../data/mapillary_exm/validation/images/dejslqZtqGZ53SiXTInqug.jpg', author="hugging_face", model=artifact, crop_n_layers=1, crop_n_points_downscale_factor=1.2)
        outputs = model(
            file_path,
            points_per_crop=kwargs["pts_per_side"],
            points_per_batch=kwargs["pts_per_batch"],
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crops_n_layers=kwargs["crop_n_layers"],
            crop_n_points_downscale_factor=kwargs["crop_n_points_downscale_factor"],
            filter_mask=filter_mask
        )

        # as in SAM, the sorting of masks actually matter
        # a, b = masks, sorted(masks, key=lambda x: x.sum(), reverse=True)
        # print(a[0].sum(), b[0].sum())
        # print(all((x==y).all() for x, y in zip(a, b)))
        if "grid_points" in outputs:
            return {
                "masks": sorted(outputs['masks'], key=lambda x: x.sum(), reverse=True), 
                "grid_points": outputs["grid_points"]
            }
        else:
            return sorted(outputs['masks'], key=lambda x: x.sum(), reverse=True)

def load_domain(name, rank=0):
    """
    load domain seg

    Args:
        name: str, name of domain model
        rank: int, in our ddp framework (1 machine * 8 GPUs) this is the device id
        dp (deprecated) : bool, DataParallel in PyTorch (1) model is copied to GPUs, data in the batch is also split and copied; (2) not suitable in Mapillary because images are of different shape
    """

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if name not in ['mask2former', 'deeplab']:
        logger.error(f"[Empty] Domain - {name}is not implemented yet")
        return None

    if name == 'mask2former':
        seg_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
        seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
        seg_model.to(device)

        artifact = {'processor':seg_processor, 'model':seg_model}

    else: # name == 'deeplab':
        seg_model = init_model(
            os.path.join(MMSEG_DIR, "configs/deeplabv3plus", "deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280.py"),
            os.path.join(MMSEG_DIR, "ckpts", "deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280_20230301_110504-655f8e43.pth"),
            device=device)

        if device.type == "cpu":
            seg_model = revert_sync_batchnorm(seg_model)

        artifact = {'model': seg_model}

    logger.info(f"[Load] Domain - {name} to {device}")

    return artifact

def infer_domain(img_input, model_name, artifact, rank):
    """
    still need to pass rank/device because intermediate outputs need to be loaded to device

    Args:
        img_input: str (i.e., path) or PIL object
        model_name : str, no need to pass author because 1-1 relationship
            (1) mask2former - 'hugging-face'
            (2) deeplab - 'mmseg'
        artifact: dict, containing model and processor, depending on the "mode"
            hugging-face: processor + model
            mmseg: model
        rank : int, GPU which device to load to
        [deprecated] save_logit: bool, whether to stop and save logits --> always return logits

    Returns:
        logits, np.array or tensor, so that other research ideas can also use results from here
    """

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    assert model_name in ['mask2former', 'deeplab']
    if model_name == 'mask2former':
        if isinstance(img_input, str):
            img_input = Image.open(img_input)

        inputs = artifact['processor'](images=img_input, return_tensors="pt").to(device)
        outputs = artifact['model'](**inputs)

        logits = utils.mask2former_postprocess(outputs.class_queries_logits, outputs.masks_queries_logits, target_sizes=[img_input.size[::-1]])[0]

    else: # 'deeplab'
        result = inference_model(artifact['model'], img_input)
        logits = result.seg_logits.data
        # label = logits.argmax(dim=0, keepdim=True)
        # prob = logits.softmax(dim=0)

    logger.info(f"[Infer] Logits produced on {device}")

    return logits

def overlay_sam_masks(
    new_mask_list: List[Any],
    author: str,
    existing_mask_np: Optional[np.array] = None,
) -> np.array:
    """
    Process all SAM masks in pre-defined order (from largest to smallest) and let them overlay.
    Only record id of the last mask that owned the pixel.

    Returns:
        last_edited_mask_per_pixel_np: np array, id of the last mask the owned the pixel
        len(sam_masks): int, total number of valid masks
    """

    for i, msk in tqdm(enumerate(new_mask_list), total=len(new_mask_list)):
        # iterate through every SAM mask
        if author == 'facebook':
            valid_mask_np = mask_utils.decode(msk['segmentation'])
        else:
            # FB version uses postprocess_small_region() to take care of hole or island
            # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/automatic_mask_generator.py#L324
            # HF version doesn't seem to have one, but we can manually copy FB's over
            if msk.sum() <= 100:
                continue
            if isinstance(msk, torch.Tensor):
                valid_mask_np = msk.cpu().numpy()
            else:
                valid_mask_np = msk
        
        valid_mask_np_bool = valid_mask_np.astype(bool)

        if i == 0: # initiate
            H, W = valid_mask_np_bool.shape
            if existing_mask_np is None:
                canvas_np = np.ones((H, W), dtype=np.int32) * (-1) 
            else:
                canvas_np = existing_mask_np.copy()

        if existing_mask_np is None:
            canvas_np[valid_mask_np_bool] = i
        else:
            canvas_np[valid_mask_np_bool & (existing_mask_np == -1)] = i + existing_mask_np.max() + 1

    return canvas_np

def find_regions_via_connected_components(multi_class_np) -> List[np.array]:
    """
    Collect all components from the overlayed output.

    alternative approaches
    (1) cv2.connectedComponents
    (2) from skimage.measure import label

    Returns:
        component_mask_list: list, list of np array containing binary mask for each component
    """
    # 0 is background
    component_mask_list = []
    for class_id in range(0, multi_class_np.max() + 1):
        binary_img = multi_class_np == class_id
        num_labels, labels_im = cv2.connectedComponents(binary_img.astype(np.uint8), connectivity=8)

        for component_id in range(1, num_labels):
            mask_np_bool = (labels_im == component_id).astype(np.uint8).astype(bool)
            component_mask_list.append(mask_np_bool)

    return component_mask_list

def extract_local_uncertainty_percentile_per_sam_mask(
    sam_masks,
    uncertainty_np,
    author,
    rank=0,
):
    """Extract local uncertainty per mask.

    Usage
    (0) from src import core; import os; import numpy as np
    (1) masks = core.infer_sam(
                '../data/mapillary_exm/validation/images/KnhdzVvaMLeXMpJvrg4XFQ.jpg',
                'hugging-face',
                core.load_sam_hf(name='vit_b'),
                pts_per_side=16,
                pts_per_batch=32,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                uncertainty_file="",
            )
    (2) uncertainty_np = np.load(
                    os.path.join(
                        "../results/",
                        "mask2former_ten",
                        "validation",
                        "KnhdzVvaMLeXMpJvrg4XFQ_mut.npy"
                    ))
    (3) core.extract_local_uncertainty_by_sam_mask(masks, uncertainty_np, 'hugging-face')
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    uncertainty_tensor = uncertainty_np
    local_uncertain_percentile_np = uncertainty_tensor.copy()
    res_list = []

    for msk in sam_masks:
        # iterate through every SAM mask
        if author == 'facebook':
            valid_mask = torch.tensor(mask_utils.decode(msk['segmentation'])).bool().to(device)
        else:
            # FB version uses postprocess_small_region() to take care of hole or island
            # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/automatic_mask_generator.py#L324
            # HF version doesn't seem to have one, but we can manually copy FB's over
            if msk.sum() <= 100:
                continue
            valid_mask = msk

        # extract local percentiles
        # for pixels where valid_mask == True, convert to percentiles among themselves
        # for pixels where valid_mask == False, remain unchanged
        local_uncertain_percentile_np = convert_masked_region_to_local_percentile(local_uncertain_percentile_np, valid_mask)
        res_list.append(local_uncertain_percentile_np)

    local_uncertain_percentile_np = np.round(
        local_uncertain_percentile_np,
        3
    )

    return local_uncertain_percentile_np, res_list

def domain_vote_per_region(
    overlayed_mask_np: Optional[np.array],
    sam_masks: List[Any],
    domain_pred: torch.Tensor,
    uncertainty_np: np.array,
    uncertainty_strategy: str,
    author: str,
    rank: int = 0,
) -> Tuple[torch.Tensor, np.array, np.array]:
    """Overlay SAM masks and then vote per region.

    Args:
        sam_masks : list of torch.tensor, each tensor is binary
        domain_pred : torch.tensor, pixel-level classification
        uncertainty_input : np.array
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # merged_mask = domain_pred.clone()
    merged_mask_tensor = torch.ones_like(domain_pred).to(device) * (-1)

    # local uncertainty np array
    local_uncertain_percentile_np: np.array = uncertainty_np.copy()

    # overlay all masks
    if overlayed_mask_np is None:
        overlayed_mask_np = overlay_sam_masks(sam_masks, author=author)
    component_mask_list = find_regions_via_connected_components(overlayed_mask_np)  # List[np.ndarray]

    # Fill local percentiles for *all* components at once (pure NumPy, vectorised)
    local_uncertain_percentile_np = fill_local_percentiles_batch(
        local_uncertain_percentile_np,
        component_mask_list,
    )

    # Build label image
    H, W = overlayed_mask_np.shape
    label_img = np.zeros_like(overlayed_mask_np, dtype=np.int32)
    for idx, comp_mask in enumerate(component_mask_list, 1):
        label_img[comp_mask] = idx

    # Vectorised majority vote in one np.bincount
    domain_pred_np = domain_pred.cpu().numpy()
    num_classes = int(domain_pred_np.max()) + 1
    majority_per_component = _vote_components_vectorized(
        label_img,
        local_uncertain_percentile_np,
        domain_pred_np,
        num_classes,
        uncertainty_strategy,
    )

    merged_mask_np = majority_per_component[label_img]
    merged_mask_tensor = torch.from_numpy(merged_mask_np).to(device)

    # areas not covered by any mask: use domain_pred default
    # the following two approaches should produce the same results
    # merged_mask[overlayed_mask == -1] = domain_pred[overlayed_mask == -1]
    covered_merge_tensor = merged_mask_tensor.clone()
    full_merge_tensor = merged_mask_tensor.clone()
    full_merge_tensor[full_merge_tensor == -1] = domain_pred[full_merge_tensor == -1]

    return covered_merge_tensor, full_merge_tensor, local_uncertain_percentile_np, overlayed_mask_np


def eval_mask_list(
    pred_list, # list[ndarray] | list[str]
    gt_list=['./data/mapillary/labels/_77MfvbukddTOIEDA9Tb5Q.png'], # list[ndarray] | list[str]
    include_index_list=[65]
):
    """Evaluate produced list of masks against ground truth PNG files.

    Usage:
    (1) core.eval_mask_list(
            [
                '../results/mask2former_one/validation/npy/labl/KnhdzVvaMLeXMpJvrg4XFQ.npy',
                '../results/mask2former_one/validation/npy/labl/ODbkOs1GqH1WP4Ua1gJdyA.npy',
                '../results/mask2former_one/validation/npy/labl/6UJguLBTFG4Kb6L9p3M2yA.npy',
                '../results/mask2former_one/validation/npy/labl/RDClvieQTUVfSvdrteZmPA.npy',
            ], # predicted classes in npy
            [
                './data/mapillary_v1/validation/labels/KnhdzVvaMLeXMpJvrg4XFQ.png',
                './data/mapillary_v1/validation/labels/ODbkOs1GqH1WP4Ua1gJdyA.png',
                './data/mapillary_v1/validation/labels/6UJguLBTFG4Kb6L9p3M2yA.png',
                './data/mapillary_v1/validation/labels/RDClvieQTUVfSvdrteZmPA.png',
            ], # ground truth in PNG
            include_index_list=[13]
        )

    merge dictionaries of different artifacts:
        {*domain_res, *merge_res}

    Args:
        pred_list : list, paths to npy files containing predicted classes
        gt_list : list, paths to ground truth png labels
        include_index_list: list, index of allowed categories for comparison
    """
    raw_res = metrics.eval_metrics(
        results=pred_list,
        gt_seg_maps=gt_list,
        num_classes=65,
        include_index_list=include_index_list,
    )

    res = {}
    # copy the average and mean metrics
    for item in raw_res:
        if item[0] in ['a', 'm']: # average and mean metrics
            res[item] = np.round(raw_res[item], 3)

    for i in range(len(MMLAB_CONFIG['classes'])):
        if i in include_index_list:
            name = MMLAB_CONFIG['classes'][i]
            res[name] = {}
            for item in ['Acc', 'IoU', 'Dice', 'Fscore', 'Precision', 'Recall']:
                res[name][item] = round(raw_res[item][i], 3)

    logger.info(f"[Eval] Resulst calculated for {len(include_index_list)} classes")
    return res


def convert_masked_region_to_local_percentile(arr: np.array, mask_np_bool: np.array) -> np.array:
    """
    Returns a new array the same shape as `arr`:
      - For elements where `mask == True`, replace them with
        their percentile rank (0..100) among just those masked elements.
      - For elements where `mask == False`, keep them as-is (or set to zero).
    """

    # Initialize output as a copy or zeros
    # out = np.zeros_like(arr, dtype=float)
    # OR if you prefer to preserve original unmasked data:
    out = arr.copy().astype(float)

    mask_np_bool = mask_np_bool.astype(bool)

    # Extract only the values in the masked region
    masked_vals = arr[mask_np_bool]

    # Rank those values; ranks go from 1 to N
    ranks = rankdata(masked_vals, method='ordinal') # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html

    # Convert rank to 0..100 percentile
    N = len(masked_vals)
    percentiles = (ranks - 1) / (N - 1) * 100 if N > 1 else np.zeros_like(ranks)

    # Place the percentile values back into `out` only where mask is True
    out[mask_np_bool] = percentiles

    return out

# -----------------------------------------------------
# Vectorised local-percentile computation (pure NumPy)
# -----------------------------------------------------


def fill_local_percentiles_batch(
    arr: np.ndarray,
    component_masks: List[np.ndarray],
) -> np.ndarray:
    """Convert *all* masked regions in *arr* to their 0-100 local percentile.

    Parameters
    ----------
    arr : np.ndarray (H, W)
        The uncertainty / score map that will be overwritten *in-place* with
        percentile values.
    component_masks : List[np.ndarray]
        List of boolean masks (same spatial shape) for every connected
        component as returned by `find_regions_via_connected_components`.

    Returns
    -------
    np.ndarray
        Reference to the same input array for convenience.
    """

    if len(component_masks) == 0:
        return arr

    # Flatten for easier indexing
    H, W = arr.shape
    flat_vals = arr.ravel()
    label_flat = np.zeros_like(flat_vals, dtype=np.int32)

    # Assign a unique label id (starting at 1) per component
    for idx, m in enumerate(component_masks, 1):
        if not m.any():  # skip empty masks (shouldn't happen, but safe)
            continue
        label_flat[m.ravel()] = idx

    # Vectorised ranking: sort by (label, value)
    order = np.lexsort((flat_vals, label_flat))  # C-level, very fast
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)

    # Pixels per label
    counts = np.bincount(label_flat)

    # For each label, the starting index in the *order* array
    # offsets_mapping[label] = starting rank for that label (background 0 has 0)
    offsets_mapping = np.cumsum(counts) - counts  # length = max_label + 1

    # Map per-pixel offset via label lookup
    offsets_per_pixel = offsets_mapping[label_flat]

    # Percentile = (rank within component) / (size-1) * 100
    denom = np.maximum(1, counts[label_flat] - 1)  # avoid divide-by-zero

    percent = (ranks - offsets_per_pixel) / denom * 100.0

    # Write back only for pixels belonging to a component (label > 0)
    flat_vals[label_flat > 0] = percent[label_flat > 0]

    return arr.reshape(H, W)

# ===============================================================
# Helper for parallel majority-vote computation per component
# ===============================================================


def _vote_single_component(
    component_mask_np_bool: np.ndarray,
    local_percent_np: np.ndarray,
    domain_pred_np: np.ndarray,
    uncertainty_strategy: str,
) -> Tuple[np.ndarray, int]:
    """Return (mask, majority_class_int) for one connected component."""

    comp_bool = component_mask_np_bool.astype(bool)

    if 'relative' in uncertainty_strategy:
        rng = uncertainty_strategy.split("relative_")[1]
        low, high = (float(x) * 100 for x in rng.split('-'))

        band = comp_bool & (local_percent_np >= low) & (local_percent_np <= high)
        comp_use = band if band.sum() > 10 else comp_bool
    else:
        comp_use = comp_bool

    # majority class among the selected pixels
    candidate_classes = domain_pred_np[comp_use]
    if candidate_classes.size == 0:
        return comp_bool, -1  # should not happen

    class_id_int = np.bincount(candidate_classes.flatten()).argmax()

    return comp_bool, int(class_id_int)

# ===============================================================
# Vectorised majority vote using a single NumPy bincount
# ===============================================================


def _vote_components_vectorized(
    label_img: np.ndarray,
    local_percent_np: np.ndarray,
    domain_pred_np: np.ndarray,
    num_classes: int,
    uncertainty_strategy: str,
) -> np.ndarray:
    """Return 1-D array `majority_per_component` (length = max_label+1)."""

    label_flat = label_img.ravel()
    pred_flat = domain_pred_np.ravel()

    n_labels = label_flat.max() + 1  # includes background (0)

    # Decide which pixels participate in the vote
    if 'relative' in uncertainty_strategy:
        rng = uncertainty_strategy.split("relative_")[1]
        low, high = (float(x) * 100 for x in rng.split('-'))

        within_band = (
            (local_percent_np.ravel() >= low) & (local_percent_np.ravel() <= high)
        )

        # per-component counts
        counts_total = np.bincount(label_flat, minlength=n_labels)
        counts_band = np.bincount(label_flat, weights=within_band.astype(np.int32), minlength=n_labels)

        use_band = counts_band >= 10

        participate_mask = (
            (use_band[label_flat] & within_band) |
            (~use_band[label_flat] & (label_flat > 0))
        )
    else:
        participate_mask = label_flat > 0  # all component pixels

    chosen_labels = label_flat[participate_mask]
    chosen_classes = pred_flat[participate_mask]

    # Combined index for bincount
    joined = chosen_labels * num_classes + chosen_classes
    hist = np.bincount(joined, minlength=n_labels * num_classes)
    hist = hist.reshape(n_labels, num_classes)

    majority = hist.argmax(axis=1)
    majority[0] = -1  # background stays -1

    return majority.astype(np.int64)

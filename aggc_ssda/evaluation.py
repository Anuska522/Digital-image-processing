"""
Module 5: Binary Change Map Generation and Evaluation

Implements:
1. Binary change map rendering and connected components morphological profiling.
2. High-visibility translucent overlay generation atop multitemporal satellite scenes.
3. Rigorous quantitative accuracy metrics (Precision, Recall, F1-Score, Overall Accuracy,
   Commission & Omission errors, Cohen's Kappa, IoU) strictly computed ONLY when valid
   ground-truth masks are provided.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def generate_change_overlay(
    base_img: np.ndarray,
    binary_change_map: np.ndarray,
    overlay_color: Tuple[int, int, int] = (255, 45, 85),  # Vibrant Crimson / Coral
    alpha: float = 0.45,
    draw_contours: bool = True,
    contour_color: Tuple[int, int, int] = (255, 230, 0),  # Bright Neon Amber
    contour_thickness: int = 1
) -> np.ndarray:
    """
    Generates an overlay highlighting detected change regions atop the satellite image.
    
    Args:
        base_img: Satellite image [H, W, 3] or [H, W], float [0, 1] or uint8 [0, 255].
        binary_change_map: Binary mask [H, W] with {0, 1}.
        overlay_color: RGB tuple for change highlight color.
        alpha: Blending transparency opacity (0.0 to 1.0).
        draw_contours: Whether to outline change boundary contours.
        contour_color: RGB tuple for boundary outlines.
        contour_thickness: Line thickness for contours.
        
    Returns:
        RGB uint8 image [H, W, 3] in range [0, 255].
    """
    # Normalize base image to uint8 RGB
    if base_img.dtype != np.uint8:
        if base_img.max() <= 1.0:
            base_u8 = np.clip(base_img * 255.0, 0, 255).astype(np.uint8)
        else:
            base_u8 = np.clip(base_img, 0, 255).astype(np.uint8)
    else:
        base_u8 = base_img.copy()

    if base_u8.ndim == 2:
        base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    elif base_u8.shape[2] == 1:
        base_rgb = cv2.cvtColor(base_u8[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif base_u8.shape[2] >= 3:
        base_rgb = base_u8[:, :, :3]
    else:
        base_rgb = np.repeat(base_u8[:, :, np.newaxis], 3, axis=2)

    h, w = base_rgb.shape[:2]
    mask = (cv2.resize(binary_change_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)

    # Color mask overlay
    overlay = base_rgb.copy()
    color_tile = np.zeros_like(base_rgb)
    color_tile[:] = overlay_color

    # Alpha blend inside mask
    overlay[mask] = (alpha * color_tile[mask] + (1.0 - alpha) * base_rgb[mask]).astype(np.uint8)

    # Draw sharp boundary contours if requested
    if draw_contours and np.any(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness)

    return overlay


def analyze_morphological_patches(binary_change_map: np.ndarray) -> Dict[str, Any]:
    """
    Computes connected component statistics for the binary change map.
    """
    mask_u8 = (binary_change_map > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    # Label 0 is background
    patch_count = num_labels - 1
    if patch_count > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        min_area = int(np.min(areas))
        max_area = int(np.max(areas))
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
    else:
        min_area = max_area = 0
        mean_area = median_area = 0.0

    return {
        "patch_count": patch_count,
        "min_patch_pixels": min_area,
        "max_patch_pixels": max_area,
        "mean_patch_pixels": round(mean_area, 2),
        "median_patch_pixels": round(median_area, 2),
    }


def evaluate_change_detection(
    predicted_change_map: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates change detection accuracy against ground truth benchmarks.
    
    If ground_truth_mask is None, returns observational summary statistics
    and clearly flags that ground truth validation was not evaluated.
    Never fabricates metrics.
    
    Args:
        predicted_change_map: Binary predicted change array [H, W] with {0, 1}.
        ground_truth_mask: Optional reference binary ground-truth array [H, W].
        
    Returns:
        Comprehensive results dictionary with metrics, confusion matrix, and morphological stats.
    """
    pred_bin = (predicted_change_map > 0).astype(np.uint8)
    total_pixels = int(pred_bin.size)
    changed_pixels = int(np.sum(pred_bin))
    unchanged_pixels = total_pixels - changed_pixels
    change_ratio = float(changed_pixels / total_pixels)

    morph_stats = analyze_morphological_patches(pred_bin)

    base_results: Dict[str, Any] = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "change_percentage": round(change_ratio * 100.0, 3),
        "morphology": morph_stats,
        "has_ground_truth": False,
    }

    if ground_truth_mask is None:
        base_results["message"] = "Observational statistics only. Upload ground truth mask to compute validation metrics."
        return base_results

    # Align ground truth mask dimensions if needed
    h, w = pred_bin.shape[:2]
    gt_raw = ground_truth_mask
    if gt_raw.ndim == 3:
        # If multi-channel, check if it's RGB mask
        gt_raw = np.mean(gt_raw, axis=-1)
    if gt_raw.shape[:2] != (h, w):
        gt_raw = cv2.resize(gt_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarize ground truth: threshold at half maximum or 0.5
    gt_thresh = 127 if gt_raw.max() > 1.0 else 0.5
    gt_bin = (gt_raw >= gt_thresh).astype(np.uint8)

    # Confusion matrix calculations
    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))

    gt_positive = tp + fn
    gt_negative = tn + fp
    pred_positive = tp + fp
    pred_negative = tn + fn

    eps = 1e-10

    # Accuracy Metrics
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    oa = float((tp + tn) / total_pixels) if total_pixels > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # Remote Sensing Error Metrics:
    # Commission Error (False Alarm in detections): FP / (TP + FP) = 1 - Precision
    commission_error = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    # Omission Error (Missed true changes): FN / (TP + FN) = 1 - Recall
    omission_error = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    # False Alarm Rate relative to true unchanged: FP / (FP + TN)
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Cohen's Kappa Coefficient
    p_observed = oa
    p_expected = ((pred_positive * gt_positive) + (pred_negative * gt_negative)) / (total_pixels ** 2 + eps)
    if (1.0 - p_expected) > eps:
        kappa = float((p_observed - p_expected) / (1.0 - p_expected))
    else:
        kappa = 1.0 if p_observed == 1.0 else 0.0

    base_results.update({
        "has_ground_truth": True,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "metrics": {
            "overall_accuracy": round(oa * 100.0, 3),
            "f1_score": round(f1_score * 100.0, 3),
            "precision": round(precision * 100.0, 3),
            "recall": round(recall * 100.0, 3),
            "iou": round(iou * 100.0, 3),
            "commission_error": round(commission_error * 100.0, 3),
            "omission_error": round(omission_error * 100.0, 3),
            "false_alarm_rate": round(false_alarm_rate * 100.0, 3),
            "cohens_kappa": round(kappa, 4),
        },
        "ground_truth_stats": {
            "true_changed_pixels": gt_positive,
            "true_unchanged_pixels": gt_negative,
            "true_change_percentage": round((gt_positive / total_pixels) * 100.0, 3),
        }
    })

    return base_results

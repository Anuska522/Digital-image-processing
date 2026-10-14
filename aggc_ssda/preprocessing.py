"""
Module 1: Multitemporal Image Preprocessing and Radiometric Calibration

Implements:
1. Compatibility and dimension verification for multitemporal acquisitions.
2. Sub-pixel spatial co-registration (ECC / ORB Homography).
3. Relative Radiometric Normalization (RRN) via Pseudo-Invariant Features (PIF)
   and histogram matching to eliminate solar illumination and atmospheric discrepancies.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def validate_image_compatibility(
    img1: np.ndarray,
    img2: np.ndarray
) -> Dict[str, Any]:
    """
    Validates dimensional and channel compatibility between multitemporal acquisitions.
    
    Args:
        img1: First temporal acquisition (T1) array.
        img2: Second temporal acquisition (T2) array.
        
    Returns:
        Dictionary containing validation status, dimensions, channels, and diagnostics.
    """
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        raise TypeError("Both input images must be NumPy ndarrays.")

    shape1 = img1.shape
    shape2 = img2.shape

    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    c1 = shape1[2] if img1.ndim == 3 else 1
    c2 = shape2[2] if img2.ndim == 3 else 1

    dimension_match = (h1 == h2) and (w1 == w2)
    channel_match = (c1 == c2)

    return {
        "valid": bool(dimension_match and channel_match),
        "shape_t1": shape1,
        "shape_t2": shape2,
        "height_t1": h1,
        "width_t1": w1,
        "channels_t1": c1,
        "channels_t2": c2,
        "dimension_match": dimension_match,
        "channel_match": channel_match,
        "dtype_t1": str(img1.dtype),
        "dtype_t2": str(img2.dtype),
    }


def to_float32_normalized(img: np.ndarray) -> np.ndarray:
    """
    Normalizes image array to float32 range [0.0, 1.0].
    """
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        if arr.max() <= 255.0:
            arr /= 255.0
        elif arr.max() <= 65535.0:
            arr /= 65535.0
        else:
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val)
    return np.clip(arr, 0.0, 1.0)


def subpixel_coregistration(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    warp_mode: int = cv2.MOTION_TRANSLATION,
    max_iterations: int = 100,
    termination_eps: float = 1e-5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies sub-pixel spatial co-registration between multitemporal acquisitions.
    Uses OpenCV Enhanced Correlation Coefficient (ECC) maximization, targeting
    residual misalignment < 0.2 pixels.
    
    Args:
        ref_img: Reference image (T1) [H, W, C] or [H, W].
        target_img: Target image to be aligned (T2).
        warp_mode: OpenCV warp mode (MOTION_TRANSLATION or MOTION_EUCLIDEAN).
        max_iterations: Maximum iterations for ECC optimization.
        termination_eps: Termination threshold for correlation coefficient delta.
        
    Returns:
        Tuple of (aligned_target_img, registration_metadata).
    """
    h_ref, w_ref = ref_img.shape[:2]
    h_tgt, w_tgt = target_img.shape[:2]

    # Resize target if initial dimensions differ
    if (h_ref != h_tgt) or (w_ref != w_tgt):
        if target_img.ndim == 3 and target_img.shape[2] > 4:
            target_resized = np.stack([
                cv2.resize(target_img[:, :, c], (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
                for c in range(target_img.shape[2])
            ], axis=-1)
        else:
            target_resized = cv2.resize(target_img, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
    else:
        target_resized = target_img.copy()

    # Convert to single-channel 8-bit for ECC calculation
    if ref_img.ndim == 3 and ref_img.shape[2] == 3:
        ref_gray = cv2.cvtColor((to_float32_normalized(ref_img) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor((to_float32_normalized(target_resized) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif ref_img.ndim == 3 and ref_img.shape[2] >= 4:
        ref_gray = (to_float32_normalized(ref_img).mean(axis=-1) * 255).astype(np.uint8)
        tgt_gray = (to_float32_normalized(target_resized).mean(axis=-1) * 255).astype(np.uint8)
    elif ref_img.ndim == 3:
        ref_gray = (to_float32_normalized(ref_img[:, :, 0]) * 255).astype(np.uint8)
        tgt_gray = (to_float32_normalized(target_resized[:, :, 0]) * 255).astype(np.uint8)
    else:
        ref_gray = (to_float32_normalized(ref_img) * 255).astype(np.uint8)
        tgt_gray = (to_float32_normalized(target_resized) * 255).astype(np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, tgt_gray, warp_matrix, warp_mode, criteria)
        
        # Warp target image (defensive per-channel handling for >4 band multispectral rasters)
        if target_resized.ndim == 3 and target_resized.shape[2] > 4:
            ch_aligned = [
                cv2.warpAffine(
                    target_resized[:, :, c],
                    warp_matrix,
                    (w_ref, h_ref),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )
                for c in range(target_resized.shape[2])
            ]
            aligned = np.stack(ch_aligned, axis=-1)
        else:
            aligned = cv2.warpAffine(
                target_resized,
                warp_matrix,
                (w_ref, h_ref),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
        
        tx = float(warp_matrix[0, 2])
        ty = float(warp_matrix[1, 2])
        displacement = float(np.sqrt(tx**2 + ty**2))
        
        metadata = {
            "success": True,
            "method": "ECC Sub-Pixel Maximization",
            "warp_matrix": warp_matrix.tolist(),
            "dx_subpixel": tx,
            "dy_subpixel": ty,
            "net_displacement_pixels": displacement,
            "subpixel_accuracy_achieved": bool(displacement < 1.0)
        }
        return aligned, metadata

    except cv2.error as e:
        # Fallback to feature-based ORB alignment if ECC fails
        aligned, fb_meta = _orb_feature_alignment(ref_img, target_resized)
        if fb_meta["success"]:
            fb_meta["method"] = "ORB Feature Matching Fallback"
            return aligned, fb_meta
        
        # If both fail, return original with informative log
        return target_resized, {
            "success": False,
            "method": "Identity (Co-registration converged or failed)",
            "error": str(e),
            "net_displacement_pixels": 0.0
        }


def _orb_feature_alignment(ref_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ORB keypoint matching fallback for co-registration."""
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        if ref_img.ndim == 3 and ref_img.shape[2] == 3:
            ref_gray = cv2.cvtColor((to_float32_normalized(ref_img) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif ref_img.ndim == 3:
            ref_gray = (to_float32_normalized(ref_img).mean(axis=-1) * 255).astype(np.uint8)
        else:
            ref_gray = (to_float32_normalized(ref_img) * 255).astype(np.uint8)

        if target_img.ndim == 3 and target_img.shape[2] == 3:
            tgt_gray = cv2.cvtColor((to_float32_normalized(target_img) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif target_img.ndim == 3:
            tgt_gray = (to_float32_normalized(target_img).mean(axis=-1) * 255).astype(np.uint8)
        else:
            tgt_gray = (to_float32_normalized(target_img) * 255).astype(np.uint8)
        
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(tgt_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return target_img, {"success": False, "reason": "Insufficient ORB features"}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 4:
            return target_img, {"success": False, "reason": "Insufficient matched features"}

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
        if homography is None:
            return target_img, {"success": False, "reason": "Homography computation failed"}

        h, w = ref_img.shape[:2]
        if target_img.ndim == 3 and target_img.shape[2] > 4:
            aligned_channels = [
                cv2.warpPerspective(target_img[:, :, c], homography, (w, h), borderMode=cv2.BORDER_REFLECT)
                for c in range(target_img.shape[2])
            ]
            aligned = np.stack(aligned_channels, axis=-1)
        else:
            aligned = cv2.warpPerspective(target_img, homography, (w, h), borderMode=cv2.BORDER_REFLECT)
        return aligned, {
            "success": True,
            "inlier_matches": int(np.sum(mask)) if mask is not None else 0
        }
    except Exception as exc:
        return target_img, {"success": False, "reason": str(exc)}


def relative_radiometric_normalization(
    ref_img: np.ndarray,
    subject_img: np.ndarray,
    method: str = "pif",
    pif_percentile: float = 15.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Relative Radiometric Normalization (RRN) using Pseudo-Invariant Features (PIFs)
    or histogram matching.
    
    Normalizes gain, offset, and atmospheric attenuation discrepancies of subject_img (T2)
    relative to reference_img (T1).
    
    Args:
        ref_img: Radiometrically stable reference acquisition (T1) [0, 1].
        subject_img: Subject acquisition to calibrate (T2) [0, 1].
        method: 'pif' (Pseudo-Invariant Features linear regression) or 'histogram_matching'.
        pif_percentile: Percentage of lowest spectral variance pixels chosen as PIFs.
        
    Returns:
        Tuple of (normalized_subject_img, normalization_metadata).
    """
    ref = to_float32_normalized(ref_img)
    subj = to_float32_normalized(subject_img)

    if method.lower() == "histogram_matching":
        from skimage.exposure import match_histograms
        matched = match_histograms(subj, ref, channel_axis=-1 if ref.ndim == 3 else None)
        return np.clip(matched, 0.0, 1.0).astype(np.float32), {
            "method": "Histogram Matching",
            "reference_mean": float(ref.mean()),
            "normalized_mean": float(matched.mean())
        }

    # PIF-based Relative Radiometric Normalization
    # 1. Identify pseudo-invariant pixels based on temporal spectral difference magnitude
    diff_mag = np.mean(np.abs(ref - subj), axis=-1 if ref.ndim == 3 else None)
    threshold = np.percentile(diff_mag, pif_percentile)
    pif_mask = diff_mag <= threshold

    # Ensure a non-trivial number of PIF samples
    min_samples = 100
    if np.sum(pif_mask) < min_samples:
        pif_mask = np.ones_like(diff_mag, dtype=bool)

    normalized_subj = np.zeros_like(subj, dtype=np.float32)
    gain_list = []
    offset_list = []

    num_channels = subj.shape[2] if subj.ndim == 3 else 1

    for c in range(num_channels):
        ref_c = ref[:, :, c] if subj.ndim == 3 else ref
        subj_c = subj[:, :, c] if subj.ndim == 3 else subj

        x_pif = subj_c[pif_mask]
        y_pif = ref_c[pif_mask]

        # Linear regression: y = gain * x + offset
        if np.std(x_pif) > 1e-6:
            poly = np.polyfit(x_pif, y_pif, 1)
            gain = float(poly[0])
            offset = float(poly[1])
        else:
            gain = 1.0
            offset = 0.0

        gain_list.append(gain)
        offset_list.append(offset)

        norm_band = gain * subj_c + offset
        if subj.ndim == 3:
            normalized_subj[:, :, c] = norm_band
        else:
            normalized_subj = norm_band

    normalized_subj = np.clip(normalized_subj, 0.0, 1.0)

    metadata = {
        "method": "Pseudo-Invariant Features (PIF) Linear Regression",
        "pif_pixel_count": int(np.sum(pif_mask)),
        "pif_percentage": float((np.sum(pif_mask) / pif_mask.size) * 100.0),
        "gains": gain_list,
        "offsets": offset_list,
        "pre_norm_subj_mean": float(subj.mean()),
        "post_norm_subj_mean": float(normalized_subj.mean()),
        "ref_mean": float(ref.mean())
    }

    return normalized_subj, metadata


def preprocess_multitemporal_pair(
    t1_img: np.ndarray,
    t2_img: np.ndarray,
    do_coregistration: bool = True,
    do_radiometric_norm: bool = True,
    rrn_method: str = "pif"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes Module 1: Preprocessing & Radiometric Calibration pipeline.
    
    Args:
        t1_img: Temporal image 1 (Reference baseline).
        t2_img: Temporal image 2 (Subject date).
        do_coregistration: Whether to perform sub-pixel alignment.
        do_radiometric_norm: Whether to perform RRN.
        rrn_method: 'pif' or 'histogram_matching'.
        
    Returns:
        Tuple of (preprocessed_t1, preprocessed_t2, diagnostics).
    """
    compat = validate_image_compatibility(t1_img, t2_img)
    diagnostics: Dict[str, Any] = {"compatibility": compat}

    t1_norm = to_float32_normalized(t1_img)
    t2_norm = to_float32_normalized(t2_img)

    # 1. Co-registration
    if do_coregistration:
        t2_aligned, reg_meta = subpixel_coregistration(t1_norm, t2_norm)
        diagnostics["coregistration"] = reg_meta
    else:
        # Match dimensions if needed
        h1, w1 = t1_norm.shape[:2]
        h2, w2 = t2_norm.shape[:2]
        if (h1 != h2) or (w1 != w2):
            if t2_norm.ndim == 3 and t2_norm.shape[2] > 4:
                t2_aligned = np.stack([
                    cv2.resize(t2_norm[:, :, c], (w1, h1), interpolation=cv2.INTER_LINEAR)
                    for c in range(t2_norm.shape[2])
                ], axis=-1)
            else:
                t2_aligned = cv2.resize(t2_norm, (w1, h1), interpolation=cv2.INTER_LINEAR)
        else:
            t2_aligned = t2_norm.copy()
        diagnostics["coregistration"] = {"bypassed": True}

    # 2. Radiometric Normalization
    if do_radiometric_norm:
        t2_calibrated, rrn_meta = relative_radiometric_normalization(
            ref_img=t1_norm,
            subject_img=t2_aligned,
            method=rrn_method
        )
        diagnostics["radiometric_normalization"] = rrn_meta
    else:
        t2_calibrated = t2_aligned
        diagnostics["radiometric_normalization"] = {"bypassed": True}

    return t1_norm, t2_calibrated, diagnostics

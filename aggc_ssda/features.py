"""
Module 2: Spatial-Spectral Feature Stack Extraction

Extracts the 4-dimensional spatial-spectral feature vector F(x, y):
    F(x, y) = [R_vis, NIR, TC_wetness, GLCM_entropy]^T

And computes the multitemporal feature difference vector:
    ΔF(x, y) = F_t2(x, y) - F_t1(x, y)

Gracefully handles band configurations (Multispectral vs. 3-band RGB fallback)
with explicit documentation and telemetry.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter


# Standard Tasseled Cap Wetness Coefficients for Landsat-8/9 OLI (Baig et al., 2014)
# Bands: [Blue, Green, Red, NIR, SWIR1, SWIR2]
TC_WETNESS_COEFFS_L8 = np.array([0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559], dtype=np.float32)

# 4-Band Optical Approximation (Blue, Green, Red, NIR) (Huang et al., 2002 adaptation)
TC_WETNESS_COEFFS_4BAND = np.array([-0.2828, -0.0573, -0.4984, 0.8172], dtype=np.float32)


def compute_glcm_texture_entropy(
    gray_img: np.ndarray,
    window_size: int = 7,
    num_levels: int = 16
) -> np.ndarray:
    r"""
    Computes Grey-Level Co-occurrence Matrix (GLCM) structural entropy over a local spatial window.
    
    Quantizes grayscale intensities to num_levels (e.g. 16 bins) and evaluates local co-occurrence
    probabilities P(i, j) at displacement d=(1, 0) to compute:
        GLCM_entropy = - \sum_{i,j} P(i, j) * log(P(i, j) + eps)
        
    Vectorized and sliding-window accelerated for real-time responsiveness.
    
    Args:
        gray_img: Single-channel grayscale image [0.0, 1.0].
        window_size: Local neighborhood window size (must be odd, e.g. 7).
        num_levels: Number of quantized intensity levels (e.g. 16).
        
    Returns:
        2D array of GLCM entropy values [H, W], normalized to [0.0, 1.0].
    """
    if not isinstance(gray_img, np.ndarray):
        raise TypeError("gray_img must be a NumPy ndarray.")
    if gray_img.ndim == 3:
        gray_img = gray_img[:, :, 0]
    elif gray_img.ndim != 2:
        raise ValueError(f"gray_img must be 2D, got shape {gray_img.shape}")

    h, w = gray_img.shape
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.float32)

    window_size = max(int(window_size), 1)
    num_levels = max(int(num_levels), 2)

    # Quantize to [0, num_levels - 1]
    quantized = np.clip((gray_img * num_levels).astype(np.int32), 0, num_levels - 1)
    
    # Neighbor horizontally displaced by 1 pixel (dx=1, dy=0)
    shifted = np.roll(quantized, shift=-1, axis=1)
    shifted[:, -1] = quantized[:, -1]  # fix boundary
    
    # Pair index k = quantized * num_levels + shifted (0 to num_levels^2 - 1)
    num_pairs = num_levels * num_levels
    pair_indices = quantized * num_levels + shifted
    
    # Calculate local probabilities using uniform_filter (equivalent to box counting in window)
    entropy_map = np.zeros((h, w), dtype=np.float32)
    eps = 1e-10
    
    # For efficiency and smooth local texture estimation, sample active level combinations
    # or evaluate local histogram entropy of paired transitions
    for k in range(num_pairs):
        mask_k = (pair_indices == k).astype(np.float32)
        if np.any(mask_k):
            # Local probability of pair k in window
            p_k = uniform_filter(mask_k, size=window_size, mode='reflect')
            # Add - p_k * log(p_k)
            valid = p_k > eps
            entropy_map[valid] -= p_k[valid] * np.log(p_k[valid])
            
    # Normalize entropy map to [0, 1]
    max_ent = np.log(num_pairs)
    if max_ent > 0:
        entropy_map = np.clip(entropy_map / max_ent, 0.0, 1.0)
        
    return entropy_map.astype(np.float32)


def extract_spatial_spectral_features(
    img: np.ndarray,
    band_mapping: Optional[Dict[str, int]] = None,
    glcm_window: int = 7
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Constructs the 4-dimensional spatial-spectral feature stack F(x, y):
        F(x, y) = [R_vis(x, y), NIR(x, y), TC_wetness(x, y), GLCM_entropy(x, y)]^T
        
    If multispectral bands (NIR, etc.) are unavailable (e.g. 3-channel RGB),
    applies transparent, documented fallbacks:
        - R_vis: Red channel (Band 0 / Red)
        - NIR fallback: Visible Green/Red leaf density proxy (VARI)
        - TC_wetness fallback: Visible Moisture Index proxy (B - R)/(B + R + eps)
        - GLCM_entropy: Full Grey-Level Co-occurrence Matrix structural entropy
        
    Args:
        img: Float32 image array in range [0, 1], shape [H, W, C] or [H, W].
        band_mapping: Optional dictionary mapping band names {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}.
        glcm_window: Local spatial window size for GLCM texture entropy.
        
    Returns:
        Tuple of (feature_stack [H, W, 4], metadata_dict).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a NumPy ndarray.")
    if img.size == 0:
        raise ValueError("Input image cannot be empty.")

    h, w = img.shape[:2]
    num_channels = img.shape[2] if img.ndim == 3 else 1

    # Default band mapping heuristic
    bands = band_mapping or {}
    metadata: Dict[str, Any] = {}

    if num_channels >= 4 and ("nir" in bands or num_channels >= 4):
        # Native Multispectral Mode (e.g. Sentinel-2, Landsat-8)
        red_idx = bands.get("red", 2)
        green_idx = bands.get("green", 1)
        blue_idx = bands.get("blue", 0)
        nir_idx = bands.get("nir", 3)

        for name, idx in [("red", red_idx), ("green", green_idx), ("blue", blue_idx), ("nir", nir_idx)]:
            if not (0 <= idx < num_channels):
                raise ValueError(f"Band index for '{name}' ({idx}) is out of bounds for {num_channels}-channel image.")

        r_vis = img[:, :, red_idx].astype(np.float32)
        nir = img[:, :, nir_idx].astype(np.float32)
        green = img[:, :, green_idx].astype(np.float32)
        blue = img[:, :, blue_idx].astype(np.float32)

        # Tasseled Cap Wetness computation (6-band Landsat-8 vs 4-band optical)
        if num_channels >= 6:
            swir1_idx = bands.get("swir1", 4)
            swir2_idx = bands.get("swir2", 5)
            if num_channels > max(swir1_idx, swir2_idx):
                swir1 = img[:, :, swir1_idx].astype(np.float32)
                swir2 = img[:, :, swir2_idx].astype(np.float32)
                stack_6b = np.stack([blue, green, r_vis, nir, swir1, swir2], axis=-1)
                tc_wetness = np.dot(stack_6b, TC_WETNESS_COEFFS_L8)
                metadata["tasseled_cap_model"] = "Landsat-8 6-Band OLI (Baig et al., 2014)"
            else:
                stack_4b = np.stack([blue, green, r_vis, nir], axis=-1)
                tc_wetness = np.dot(stack_4b, TC_WETNESS_COEFFS_4BAND)
                metadata["tasseled_cap_model"] = "4-Band Optical Approximation"
        else:
            stack_4b = np.stack([blue, green, r_vis, nir], axis=-1)
            tc_wetness = np.dot(stack_4b, TC_WETNESS_COEFFS_4BAND)
            metadata["tasseled_cap_model"] = "4-Band Optical Approximation"

        # Rescale tc_wetness to [0, 1] range for feature compatibility
        tc_min, tc_max = tc_wetness.min(), tc_wetness.max()
        if tc_max > tc_min:
            tc_wetness = (tc_wetness - tc_min) / (tc_max - tc_min)
        else:
            tc_wetness = np.zeros_like(tc_wetness)

        # Grayscale representation for GLCM
        gray = 0.299 * r_vis + 0.587 * green + 0.114 * blue
        glcm_entropy = compute_glcm_texture_entropy(gray, window_size=glcm_window)

        mode = "Multispectral (Native NIR & Tasseled Cap)"
        metadata["mode"] = mode
        metadata["fallback_applied"] = False
        metadata["band_assignments"] = {"red": red_idx, "green": green_idx, "blue": blue_idx, "nir": nir_idx}

    elif num_channels == 3:
        # Standard RGB Mode (Fallback Triggered)
        red_idx = bands.get("red", 0)
        green_idx = bands.get("green", 1)
        blue_idx = bands.get("blue", 2)

        for name, idx in [("red", red_idx), ("green", green_idx), ("blue", blue_idx)]:
            if not (0 <= idx < num_channels):
                raise ValueError(f"Band index for '{name}' ({idx}) is out of bounds for 3-channel image.")

        red = img[:, :, red_idx].astype(np.float32)
        green = img[:, :, green_idx].astype(np.float32)
        blue = img[:, :, blue_idx].astype(np.float32)

        r_vis = red

        # Fallback 1: Leaf structural chlorophyll proxy (VARI - Visible Atmospherically Resistant Index)
        # VARI = (Green - Red) / (Green + Red - Blue + eps)
        eps = 1e-6
        denom = green + red - blue
        denom[np.abs(denom) < eps] = eps
        vari = (green - red) / denom
        # Normalize to [0, 1]
        nir_proxy = np.clip((vari + 1.0) / 2.0, 0.0, 1.0)

        # Fallback 2: Visible Moisture / Soil Wetness proxy
        # Contrast of blue reflectance over red soil absorption
        wetness_proxy = (blue - red) / (blue + red + eps)
        wetness_proxy = np.clip((wetness_proxy + 1.0) / 2.0, 0.0, 1.0)

        # Grayscale for GLCM
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        glcm_entropy = compute_glcm_texture_entropy(gray, window_size=glcm_window)

        r_vis = np.clip(r_vis, 0.0, 1.0)
        nir = nir_proxy
        tc_wetness = wetness_proxy

        mode = "RGB Fallback Mode (NIR & Wetness approximated via calibrated visible indices)"
        metadata["mode"] = mode
        metadata["fallback_applied"] = True
        metadata["band_assignments"] = {"red": red_idx, "green": green_idx, "blue": blue_idx}
        metadata["fallback_details"] = {
            "R_vis": "Direct Red band reflectance (chlorophyll absorption)",
            "NIR": "Visible Atmospherically Resistant Index (VARI) leaf proxy",
            "TC_wetness": "Visible Moisture Gradient proxy (B-R)/(B+R)",
            "GLCM_entropy": "Full Grey-Level Co-occurrence Matrix texture entropy"
        }

    else:
        # Single-channel / Grayscale Fallback
        gray = img[:, :, 0] if img.ndim == 3 else img
        r_vis = gray.astype(np.float32)
        nir = gray.astype(np.float32)
        tc_wetness = gray.astype(np.float32)
        glcm_entropy = compute_glcm_texture_entropy(gray, window_size=glcm_window)

        mode = "Single-channel Grayscale Mode"
        metadata["mode"] = mode
        metadata["fallback_applied"] = True

    # Assemble 4D feature stack [H, W, 4]
    feature_stack = np.stack([r_vis, nir, tc_wetness, glcm_entropy], axis=-1).astype(np.float32)
    metadata["feature_stack_shape"] = feature_stack.shape

    return feature_stack, metadata


def compute_feature_difference(
    feat1: np.ndarray,
    feat2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes multitemporal feature difference vector:
        ΔF(x, y) = F_t2(x, y) - F_t1(x, y)
    and simple Euclidean spectral magnitude:
        ||ΔF(x, y)||_2
        
    Args:
        feat1: Feature stack of T1 [H, W, D].
        feat2: Feature stack of T2 [H, W, D].
        
    Returns:
        Tuple of (delta_F [H, W, D], euclidean_magnitude [H, W]).
    """
    delta_f = (feat2 - feat1).astype(np.float32)
    euclidean_mag = np.linalg.norm(delta_f, axis=-1).astype(np.float32)
    return delta_f, euclidean_mag

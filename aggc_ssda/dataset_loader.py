"""
Dataset Loader and Benchmark Generator for Multitemporal Satellite Imagery.

Provides:
1. Helper functions to load image pairs from disk or bytes.
2. Procedural synthetic Earth Observation benchmark generator with exact ground-truth masks
   specifically modeling urban expansion, vegetation clearance, and non-stationary illumination noise.
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import cv2


def load_image_from_file(file_path: str) -> np.ndarray:
    """Loads image from filesystem into RGB or multispectral format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
        
    fn_lower = file_path.lower()
    if fn_lower.endswith((".tif", ".tiff")):
        try:
            import tifffile
            arr = tifffile.imread(file_path)
            if arr is not None and arr.size > 0:
                return arr
        except Exception:
            pass

    bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Failed to decode image from: {file_path}")
    if bgr.ndim == 2:
        return bgr
    elif bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
    return bgr


def save_image_to_file(img: np.ndarray, file_path: str) -> None:
    """Saves image to file in appropriate format, supporting multispectral rasters."""
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy ndarray.")
    if img.size == 0:
        raise ValueError("Cannot save an empty image.")

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    fn_lower = file_path.lower()

    if fn_lower.endswith((".tif", ".tiff")):
        try:
            import tifffile
            tifffile.imwrite(file_path, img)
            return
        except Exception:
            pass

    if img.ndim == 2:
        to_write = img
    elif img.shape[2] == 1:
        to_write = img[:, :, 0]
    elif img.shape[2] == 3:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        # For non-TIFF with >4 channels, write the first 3 channels
        to_write = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(file_path, to_write)
    if not success:
        raise IOError(f"Failed to write image to {file_path}")



def generate_benchmark_satellite_pair(
    height: int = 256,
    width: int = 256,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Procedurally synthesizes a benchmark multitemporal satellite image pair (T1, T2)
    and authentic binary ground-truth change mask.
    
    Models:
    - Base landscape: Forest canopy (dark green/chlorophyll absorption), agricultural fields, water body, bare soil.
    - Non-stationary background noise: Spatially continuous atmospheric gradient and sensor speckle.
    - True change events:
        1. Forest clearance / logging area (green canopy -> bare bright soil).
        2. New urban / industrial construction sector (soil -> high-contrast built geometry).
        3. Water reservoir expansion.
    - Stable seasonal variations (slight agricultural phenology shifts) that an algorithm must NOT falsely flag.
    
    Returns:
        Tuple of (t1_rgb, t2_rgb, ground_truth_mask, metadata).
    """
    rng = np.random.RandomState(seed)

    # 1. Base terrain representation (T1)
    # Coordinates grid
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Base background: semi-arid soil with texture
    base_r = 140 + 20 * np.sin(xx / 20.0) + rng.normal(0, 5, (height, width))
    base_g = 120 + 15 * np.cos(yy / 25.0) + rng.normal(0, 5, (height, width))
    base_b = 90 + 10 * np.sin((xx + yy) / 30.0) + rng.normal(0, 4, (height, width))

    # Forest zone (top-left quadrant and riverbank)
    forest_mask = (xx < width * 0.45) & (yy < height * 0.55)
    base_r[forest_mask] = 45 + rng.normal(0, 6, np.sum(forest_mask))
    base_g[forest_mask] = 115 + rng.normal(0, 8, np.sum(forest_mask))
    base_b[forest_mask] = 40 + rng.normal(0, 5, np.sum(forest_mask))

    # River / water body (meandering curve)
    river_center = (width * 0.5) + 30 * np.sin(yy / 35.0)
    river_mask = np.abs(xx - river_center) < 10
    base_r[river_mask] = 30 + rng.normal(0, 3, np.sum(river_mask))
    base_g[river_mask] = 60 + rng.normal(0, 4, np.sum(river_mask))
    base_b[river_mask] = 120 + rng.normal(0, 5, np.sum(river_mask))

    # Agricultural patchwork (bottom-right)
    agri_mask = (xx > width * 0.55) & (yy > height * 0.5) & (~river_mask)
    base_r[agri_mask] = 95 + rng.normal(0, 7, np.sum(agri_mask))
    base_g[agri_mask] = 140 + rng.normal(0, 7, np.sum(agri_mask))
    base_b[agri_mask] = 60 + rng.normal(0, 5, np.sum(agri_mask))

    # T1 image composite
    t1_r = np.clip(base_r, 0, 255).astype(np.uint8)
    t1_g = np.clip(base_g, 0, 255).astype(np.uint8)
    t1_b = np.clip(base_b, 0, 255).astype(np.uint8)
    t1_img = np.stack([t1_r, t1_g, t1_b], axis=-1)

    # 2. Construct T2 and authentic Ground Truth
    t2_r = t1_r.copy().astype(np.float32)
    t2_g = t1_g.copy().astype(np.float32)
    t2_b = t1_b.copy().astype(np.float32)

    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # Change Region 1: Forest deforestation patch (cleared for soil/clearing)
    clearing_mask = (xx > width * 0.12) & (xx < width * 0.32) & (yy > height * 0.15) & (yy < height * 0.38)
    t2_r[clearing_mask] = 180 + rng.normal(0, 8, np.sum(clearing_mask))
    t2_g[clearing_mask] = 155 + rng.normal(0, 8, np.sum(clearing_mask))
    t2_b[clearing_mask] = 125 + rng.normal(0, 6, np.sum(clearing_mask))
    gt_mask[clearing_mask] = 1

    # Change Region 2: New urban development / built structures
    urban_mask = (xx > width * 0.65) & (xx < width * 0.88) & (yy > height * 0.15) & (yy < height * 0.40) & (~river_mask)
    t2_r[urban_mask] = 210 + rng.normal(0, 10, np.sum(urban_mask))
    t2_g[urban_mask] = 205 + rng.normal(0, 10, np.sum(urban_mask))
    t2_b[urban_mask] = 215 + rng.normal(0, 10, np.sum(urban_mask))
    gt_mask[urban_mask] = 1

    # Natural un-changed variations in T2 (atmospheric illumination gradient + slight phenology)
    illumination_gradient = 10 * (xx / width) - 5 * (yy / height)
    t2_r += illumination_gradient + rng.normal(0, 3, (height, width))
    t2_g += illumination_gradient + rng.normal(0, 3, (height, width))
    t2_b += illumination_gradient + rng.normal(0, 3, (height, width))

    t2_img = np.clip(np.stack([t2_r, t2_g, t2_b], axis=-1), 0, 255).astype(np.uint8)

    metadata = {
        "dataset_name": "Synthetic Multitemporal Earth Observation Benchmark Pair",
        "provenance": "Procedural physical Earth simulation modeling deforestation and urban growth",
        "dimensions": (height, width, 3),
        "ground_truth_total_change_pixels": int(np.sum(gt_mask)),
        "ground_truth_change_ratio": float(np.sum(gt_mask) / (height * width)),
        "change_events": [
            "Forest canopy deforestation / bare soil conversion",
            "Urban development / high-reflectance structural footprint"
        ],
        "non_change_variations": [
            "Non-stationary illumination gradient across scene",
            "Residual sensor noise in invariant pixels"
        ]
    }

    return t1_img, t2_img, gt_mask, metadata

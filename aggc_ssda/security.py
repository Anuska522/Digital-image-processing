"""
Security, Input Validation, and Resource Bounding for AGGC-SSDA.

Implements production-grade defensive controls:
1. Decompression bomb protection (PIL pixel ceilings).
2. Resource limits on file size and pixel dimensions.
3. Multispectral band preservation during byte decoding (TIFF, GeoTIFF, PNG, JPEG).
4. Path traversal sanitization for file loading and artifact exports.
"""

from typing import Tuple, Optional, Dict, Any
import os
import io
import numpy as np
from PIL import Image

# Enforce strict decompression bomb ceiling on PIL
Image.MAX_IMAGE_PIXELS = 100_000_000

# Configurable limits from environment with safe production defaults
MAX_UPLOAD_BYTES = int(os.environ.get("AGGC_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_IMAGE_DIMENSION = int(os.environ.get("AGGC_MAX_DIM_PIXELS", "8192"))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SecurityValidationError(ValueError):
    """Raised when an input fails security or resource bounds checks."""
    pass


def validate_file_size(file_bytes: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Verifies that uploaded byte stream does not exceed maximum permitted memory limits."""
    size = len(file_bytes)
    if size == 0:
        raise SecurityValidationError("Uploaded file is empty (0 bytes).")
    if size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        size_mb = size / (1024 * 1024)
        raise SecurityValidationError(
            f"File size ({size_mb:.2f} MB) exceeds maximum permitted limit ({max_mb:.0f} MB)."
        )


def validate_image_dimensions(height: int, width: int, max_dim: int = MAX_IMAGE_DIMENSION) -> None:
    """Guards against excessively large rasters that could cause out-of-memory denial of service."""
    if height <= 0 or width <= 0:
        raise SecurityValidationError(f"Invalid image dimensions: {width}x{height}.")
    if height > max_dim or width > max_dim:
        raise SecurityValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dim}x{max_dim} px)."
        )


def sanitize_filename(filename: str, default: str = "output.png") -> str:
    """Strips path traversal elements, slashes, and control characters from filenames."""
    if not filename:
        return default
    basename = os.path.basename(filename)
    clean = "".join(c for c in basename if c.isalnum() or c in "._- ")
    return clean if clean else default


def validate_safe_path(target_path: str, base_dir: str) -> str:
    """
    Ensures target_path resolves strictly inside base_dir, preventing path traversal attacks.
    Robust against prefix collision attacks (e.g. /app/data vs /app/database) and case sensitivity.
    """
    norm_base = os.path.normcase(os.path.abspath(base_dir))
    norm_target = os.path.normcase(os.path.abspath(target_path))
    try:
        common = os.path.commonpath([norm_base, norm_target])
        if common != norm_base:
            raise SecurityValidationError(f"Path traversal detected: {target_path} is outside {base_dir}.")
    except (ValueError, Exception):
        raise SecurityValidationError(f"Path traversal detected: {target_path} is outside {base_dir}.")
    return os.path.abspath(target_path)


def safe_decode_image_bytes(
    file_bytes: bytes,
    filename: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Safely decodes raw bytes into a validated NumPy array while preserving multispectral channels.
    
    Handles:
    - Multi-band TIFF / GeoTIFF rasters (4+ bands) via tifffile/Pillow without channel stripping.
    - Standard RGB and single-channel grayscale rasters.
    - RGBA rasters (drops alpha only when standard transparency, or preserves when requested).
    
    Returns:
        Tuple of (image_array, metadata_dict).
    """
    validate_file_size(file_bytes)
    fn_lower = (filename or "").lower()

    metadata: Dict[str, Any] = {
        "source_filename": sanitize_filename(filename) if filename else "memory_buffer",
        "file_size_bytes": len(file_bytes),
    }

    # If file is a TIFF / GeoTIFF, attempt reading with tifffile first for multispectral integrity
    if fn_lower.endswith((".tif", ".tiff")):
        try:
            import tifffile
            with io.BytesIO(file_bytes) as bio:
                arr = tifffile.imread(bio)
            if arr is not None and arr.size > 0:
                # Transpose band-first (C, H, W) to (H, W, C) for standard pipeline consumption
                if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                    arr = np.transpose(arr, (1, 2, 0))
                h, w = arr.shape[:2]
                validate_image_dimensions(h, w)
                metadata["format"] = "TIFF (Multispectral Preserved)"
                metadata["shape"] = arr.shape
                metadata["dtype"] = str(arr.dtype)
                return arr, metadata
        except Exception:
            pass  # Fall back to Pillow below

    # Standard decode using Pillow with decompression bomb guard
    try:
        with io.BytesIO(file_bytes) as bio:
            with Image.open(bio) as pil_img:
                h, w = pil_img.height, pil_img.width
                validate_image_dimensions(h, w)
                mode = pil_img.mode
                metadata["pil_mode"] = mode
                metadata["format"] = pil_img.format or "UNKNOWN"

                # Check if multi-frame / multi-band TIFF
                n_frames = getattr(pil_img, "n_frames", 1)
                if n_frames > 1 and fn_lower.endswith((".tif", ".tiff")):
                    frames = []
                    for idx in range(n_frames):
                        pil_img.seek(idx)
                        frames.append(np.array(pil_img))
                    arr = np.stack(frames, axis=-1)
                else:
                    arr = np.array(pil_img)

                # Handle color modes
                if arr.ndim == 3 and arr.shape[2] == 4 and mode == "RGBA":
                    # For standard RGBA web images, channel 4 is alpha transparency
                    metadata["alpha_stripped"] = True
                    arr = arr[:, :, :3]
                elif arr.ndim == 3 and arr.shape[2] >= 4:
                    metadata["multispectral_bands_preserved"] = arr.shape[2]

                metadata["shape"] = arr.shape
                metadata["dtype"] = str(arr.dtype)
                return arr, metadata

    except Exception as exc:
        raise SecurityValidationError(f"Failed to decode image content safely: {str(exc)}") from exc

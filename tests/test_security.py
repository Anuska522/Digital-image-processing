"""
Unit and defensive security tests for aggc_ssda.security.
"""

import os
import io
import tempfile
import pytest
import numpy as np
from PIL import Image
from aggc_ssda.security import (
    validate_file_size,
    validate_image_dimensions,
    sanitize_filename,
    validate_safe_path,
    safe_decode_image_bytes,
    SecurityValidationError,
)


def test_validate_file_size_empty():
    with pytest.raises(SecurityValidationError, match="empty"):
        validate_file_size(b"")


def test_validate_file_size_exceeded():
    with pytest.raises(SecurityValidationError, match="exceeds maximum permitted limit"):
        validate_file_size(b"X" * 100, max_bytes=50)


def test_validate_file_size_valid():
    validate_file_size(b"X" * 50, max_bytes=100)


def test_validate_image_dimensions_invalid():
    with pytest.raises(SecurityValidationError, match="Invalid image dimensions"):
        validate_image_dimensions(0, 100)
    with pytest.raises(SecurityValidationError, match="exceed maximum allowed"):
        validate_image_dimensions(9000, 100, max_dim=8192)


def test_validate_image_dimensions_valid():
    validate_image_dimensions(512, 512, max_dim=8192)


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd.png") == "passwd.png"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("my_image@#$%^&*.tif") == "my_image.tif"
    assert sanitize_filename("", default="fallback.png") == "fallback.png"


def test_validate_safe_path():
    with tempfile.TemporaryDirectory() as base_dir:
        safe = os.path.join(base_dir, "subdir", "result.png")
        unsafe = os.path.join(base_dir, "..", "secret.txt")
        
        assert validate_safe_path(safe, base_dir) == os.path.abspath(safe)
        with pytest.raises(SecurityValidationError, match="Path traversal detected"):
            validate_safe_path(unsafe, base_dir)


def test_safe_decode_image_bytes_valid_png():
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    raw_bytes = bio.getvalue()
    
    arr, meta = safe_decode_image_bytes(raw_bytes, "test.png")
    assert arr.shape == (64, 64, 3)
    assert meta["format"] == "PNG"


def test_safe_decode_image_corrupt_data():
    with pytest.raises(SecurityValidationError, match="Failed to decode"):
        safe_decode_image_bytes(b"NOT_AN_IMAGE_DATA_12345", "test.png")


def test_validate_safe_path_prefix_collision_blocked():
    with tempfile.TemporaryDirectory() as base_dir:
        # Create a directory path that has base_dir as a substring prefix but is a sibling directory
        sibling_dir = base_dir.rstrip("\\/") + "_sibling"
        target = os.path.join(sibling_dir, "file.txt")
        with pytest.raises(SecurityValidationError, match="Path traversal detected"):
            validate_safe_path(target, base_dir)


def test_safe_decode_image_bytes_geotiff_band_first():
    # Test TIFF with band-first shape (4, 32, 32)
    import tifffile
    synth_geotiff = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint8)
    bio = io.BytesIO()
    tifffile.imwrite(bio, synth_geotiff, photometric='minisblack')
    raw_bytes = bio.getvalue()

    arr, meta = safe_decode_image_bytes(raw_bytes, "scene.tif")
    # Should be transposed to (32, 32, 4)
    assert arr.shape == (32, 32, 4)
    assert "TIFF" in meta["format"]


"""
Unit tests for Module 1: Preprocessing and Radiometric Calibration.
"""

import numpy as np
import pytest
from aggc_ssda.preprocessing import (
    validate_image_compatibility,
    to_float32_normalized,
    subpixel_coregistration,
    relative_radiometric_normalization,
    preprocess_multitemporal_pair,
)


def test_validate_image_compatibility():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8)
    res = validate_image_compatibility(img1, img2)
    assert res["valid"] is True
    assert res["dimension_match"] is True
    assert res["channel_match"] is True

    # Incompatible dimensions
    img3 = np.zeros((120, 100, 3), dtype=np.uint8)
    res_incompat = validate_image_compatibility(img1, img3)
    assert res_incompat["valid"] is False
    assert res_incompat["dimension_match"] is False


def test_to_float32_normalized():
    img_u8 = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    norm = to_float32_normalized(img_u8)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert np.isclose(norm[0, 0], 0.0)
    assert np.isclose(norm[1, 0], 1.0)


def test_subpixel_coregistration():
    rng = np.random.RandomState(42)
    base = rng.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)
    # Slight shift
    target = np.roll(base, shift=1, axis=1)
    
    aligned, meta = subpixel_coregistration(base, target)
    assert aligned.shape == base.shape
    assert "method" in meta


def test_relative_radiometric_normalization():
    rng = np.random.RandomState(42)
    ref = rng.uniform(0.2, 0.8, (60, 60, 3)).astype(np.float32)
    # Subject has artificial gain and offset
    subj = np.clip(1.2 * ref + 0.1, 0.0, 1.0)

    norm_subj, meta = relative_radiometric_normalization(ref, subj, method="pif")
    assert norm_subj.shape == subj.shape
    assert meta["pif_pixel_count"] > 0
    # Mean of normalized subject should move closer to reference mean
    diff_before = abs(subj.mean() - ref.mean())
    diff_after = abs(norm_subj.mean() - ref.mean())
    assert diff_after <= diff_before


def test_preprocess_multitemporal_pair():
    t1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    t2 = np.ones((50, 50, 3), dtype=np.uint8) * 120
    t1_p, t2_p, diag = preprocess_multitemporal_pair(t1, t2)
    assert t1_p.shape == (50, 50, 3)
    assert t2_p.shape == (50, 50, 3)
    assert "compatibility" in diag
    assert "coregistration" in diag
    assert "radiometric_normalization" in diag


def test_subpixel_coregistration_different_channels():
    # Test 3-channel ref with 1-channel target
    ref_3c = np.ones((40, 40, 3), dtype=np.float32) * 0.5
    tgt_1c = np.ones((40, 40), dtype=np.float32) * 0.4
    aligned, meta = subpixel_coregistration(ref_3c, tgt_1c)
    assert aligned.shape[:2] == (40, 40)
    assert "method" in meta

    # Test 2D grayscale ref with 3-channel target
    ref_gray = np.ones((40, 40), dtype=np.float32) * 0.5
    tgt_3c = np.ones((40, 40, 3), dtype=np.float32) * 0.4
    aligned3, meta3 = subpixel_coregistration(ref_gray, tgt_3c)
    assert aligned3.shape == (40, 40, 3)
    assert "method" in meta3


def test_relative_radiometric_normalization_mismatched_channels():
    # 2D reference with 3-channel subject
    ref_2d = np.ones((30, 30), dtype=np.float32) * 0.5
    subj_3d = np.ones((30, 30, 3), dtype=np.float32) * 0.6
    norm_subj, meta = relative_radiometric_normalization(ref_2d, subj_3d, method="pif")
    assert norm_subj.shape == (30, 30, 3)
    assert meta["pif_pixel_count"] > 0

    # 4-channel subject with 3-channel reference
    ref_3c = np.ones((30, 30, 3), dtype=np.float32) * 0.5
    subj_4c = np.ones((30, 30, 4), dtype=np.float32) * 0.6
    norm_4c, meta_4c = relative_radiometric_normalization(ref_3c, subj_4c, method="pif")
    assert norm_4c.shape == (30, 30, 4)


def test_to_float32_normalized_edge_cases():
    # Empty array
    empty = np.empty((0, 0), dtype=np.float32)
    assert to_float32_normalized(empty).size == 0

    # Negative values
    neg_arr = np.array([[-10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    norm_neg = to_float32_normalized(neg_arr)
    assert norm_neg.min() >= 0.0
    assert norm_neg.max() <= 1.0

    # 16-bit range
    u16_arr = np.array([[0, 32768], [65535, 10000]], dtype=np.uint16)
    norm_u16 = to_float32_normalized(u16_arr)
    assert norm_u16.min() >= 0.0
    assert norm_u16.max() <= 1.0


def test_validate_image_compatibility_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_image_compatibility(np.zeros((0, 0)), np.zeros((10, 10)))
    with pytest.raises(TypeError, match="NumPy ndarrays"):
        validate_image_compatibility([1, 2, 3], np.zeros((10, 10)))


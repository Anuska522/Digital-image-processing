"""
Tests for native multispectral (4-band and 6-band) image processing.
Ensures that multispectral bands are preserved without degradation.
"""

import numpy as np
import pytest
from aggc_ssda.preprocessing import (
    validate_image_compatibility,
    preprocess_multitemporal_pair,
    subpixel_coregistration,
)
from aggc_ssda.features import (
    extract_spatial_spectral_features,
    compute_feature_difference,
)


def test_4band_optical_features():
    # 4-band synthetic raster: Blue, Green, Red, NIR
    h, w = 64, 64
    img4 = np.random.uniform(0.1, 0.9, (h, w, 4)).astype(np.float32)
    feat_stack, meta = extract_spatial_spectral_features(
        img4,
        band_mapping={"blue": 0, "green": 1, "red": 2, "nir": 3}
    )
    
    assert feat_stack.shape == (h, w, 4)
    assert meta["fallback_applied"] is False
    assert "Multispectral" in meta["mode"]
    assert meta["tasseled_cap_model"] == "4-Band Optical Approximation"


def test_6band_landsat8_features():
    # 6-band synthetic Landsat-8 raster: Blue, Green, Red, NIR, SWIR1, SWIR2
    h, w = 64, 64
    img6 = np.random.uniform(0.1, 0.9, (h, w, 6)).astype(np.float32)
    feat_stack, meta = extract_spatial_spectral_features(
        img6,
        band_mapping={"blue": 0, "green": 1, "red": 2, "nir": 3, "swir1": 4, "swir2": 5}
    )
    
    assert feat_stack.shape == (h, w, 4)
    assert meta["fallback_applied"] is False
    assert "Landsat-8 6-Band OLI" in meta["tasseled_cap_model"]


def test_multispectral_coregistration_preserves_channels():
    # 6-band image co-registration test
    h, w = 48, 48
    t1 = np.random.uniform(0.2, 0.8, (h, w, 6)).astype(np.float32)
    t2 = np.roll(t1, shift=2, axis=0)  # shift by 2 rows
    
    aligned, meta = subpixel_coregistration(t1, t2)
    assert aligned.shape == (h, w, 6)


def test_multispectral_pipeline_preprocessing():
    # 4-band preprocessing pipeline test
    h, w = 40, 40
    t1 = np.random.uniform(0.1, 0.8, (h, w, 4)).astype(np.float32)
    t2 = np.random.uniform(0.1, 0.8, (h, w, 4)).astype(np.float32)
    
    p_t1, p_t2, diag = preprocess_multitemporal_pair(t1, t2, do_coregistration=True, do_radiometric_norm=True)
    assert p_t1.shape == (h, w, 4)
    assert p_t2.shape == (h, w, 4)
    assert diag["compatibility"]["channels_t1"] == 4
    assert diag["compatibility"]["channels_t2"] == 4

"""
Unit tests for Module 2: Spatial-Spectral Feature Extraction.
"""

import numpy as np
import pytest
from aggc_ssda.features import (
    compute_glcm_texture_entropy,
    extract_spatial_spectral_features,
    compute_feature_difference
)


def test_compute_glcm_texture_entropy():
    gray = np.zeros((32, 32), dtype=np.float32)
    # Uniform image has low entropy
    ent_uniform = compute_glcm_texture_entropy(gray, window_size=5)
    assert ent_uniform.shape == (32, 32)
    assert ent_uniform.max() == 0.0

    # Highly textured image has positive entropy
    rng = np.random.RandomState(42)
    textured = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    ent_tex = compute_glcm_texture_entropy(textured, window_size=5)
    assert ent_tex.max() > 0.0
    assert ent_tex.max() <= 1.0


def test_extract_spatial_spectral_features_rgb():
    img_rgb = np.ones((40, 40, 3), dtype=np.float32) * 0.5
    feat, meta = extract_spatial_spectral_features(img_rgb)
    assert feat.shape == (40, 40, 4)
    assert meta["fallback_applied"] is True
    assert "R_vis" in meta["fallback_details"]
    assert "NIR" in meta["fallback_details"]
    assert "TC_wetness" in meta["fallback_details"]
    assert "GLCM_entropy" in meta["fallback_details"]


def test_extract_spatial_spectral_features_multispectral():
    img_4b = np.ones((40, 40, 4), dtype=np.float32) * 0.5
    feat, meta = extract_spatial_spectral_features(img_4b)
    assert feat.shape == (40, 40, 4)
    assert meta["fallback_applied"] is False
    assert "band_assignments" in meta


def test_compute_feature_difference():
    f1 = np.ones((20, 20, 4), dtype=np.float32) * 0.2
    f2 = np.ones((20, 20, 4), dtype=np.float32) * 0.5
    diff, mag = compute_feature_difference(f1, f2)
    assert diff.shape == (20, 20, 4)
    assert np.allclose(diff, 0.3)
    expected_mag = np.sqrt(4 * (0.3**2))
    assert np.allclose(mag, expected_mag, atol=1e-5)

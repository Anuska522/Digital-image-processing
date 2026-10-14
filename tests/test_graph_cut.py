"""
Unit tests for Module 4: Contrast-Weighted Graph-Cut Energy Minimization.
"""

import numpy as np
import pytest
from aggc_ssda.graph_cut import (
    compute_pairwise_contrast_weights,
    optimize_graph_cut_change_map,
    HAS_PYMAXFLOW,
)


def test_compute_pairwise_contrast_weights():
    feat = np.zeros((20, 20, 4), dtype=np.float32)
    # Add step edge in right half
    feat[:, 10:, :] = 1.0

    weights, sigma_sq = compute_pairwise_contrast_weights(feat, lambda_param=2.0)
    assert sigma_sq > 0
    assert "horizontal" in weights
    assert "vertical" in weights
    assert "diag_right" in weights
    assert "diag_left" in weights
    # At the step edge, contrast is high so edge penalty should be relaxed (lower weight)
    # Across smooth region, contrast is zero so weight should be close to lambda
    assert weights["horizontal"][:, 9].mean() < weights["horizontal"][:, 2].mean()


def test_optimize_graph_cut_change_map():
    h, w = 30, 30
    # Clean background in outer region, strong change in center 10x10
    d_unchanged = np.zeros((h, w), dtype=np.float32)
    d_changed = np.ones((h, w), dtype=np.float32) * 5.0

    # In center: flip unary costs to favor change
    d_unchanged[10:20, 10:20] = 5.0
    d_changed[10:20, 10:20] = 0.0

    feature_guide = np.zeros((h, w, 4), dtype=np.float32)

    # Test default solver (PyMaxflow if present, else fallback)
    change_map, meta = optimize_graph_cut_change_map(
        d_unchanged=d_unchanged,
        d_changed=d_changed,
        feature_guide=feature_guide,
        lambda_param=1.0
    )

    assert change_map.shape == (h, w)
    assert set(np.unique(change_map)).issubset({0, 1})
    assert np.all(change_map[10:20, 10:20] == 1)
    assert np.all(change_map[:5, :5] == 0)
    assert meta["changed_pixels"] == 100


def test_optimize_graph_cut_change_map_fallback():
    """Explicitly verifies the pure-NumPy fallback solver works independently of PyMaxflow."""
    h, w = 30, 30
    d_unchanged = np.zeros((h, w), dtype=np.float32)
    d_changed = np.ones((h, w), dtype=np.float32) * 5.0

    d_unchanged[10:20, 10:20] = 5.0
    d_changed[10:20, 10:20] = 0.0

    feature_guide = np.zeros((h, w, 4), dtype=np.float32)

    change_map, meta = optimize_graph_cut_change_map(
        d_unchanged=d_unchanged,
        d_changed=d_changed,
        feature_guide=feature_guide,
        lambda_param=1.0,
        force_fallback=True
    )

    assert change_map.shape == (h, w)
    assert set(np.unique(change_map)).issubset({0, 1})
    assert np.all(change_map[10:20, 10:20] == 1)
    assert np.all(change_map[:5, :5] == 0)
    assert meta["changed_pixels"] == 100
    assert "Pure-Python" in meta["solver"]


def test_optimize_graph_cut_validation_and_edge_cases():
    # Mismatched shapes
    d_u = np.zeros((10, 10), dtype=np.float32)
    d_c = np.zeros((12, 10), dtype=np.float32)
    guide = np.zeros((10, 10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        optimize_graph_cut_change_map(d_u, d_c, guide)

    # Guide spatial mismatch
    guide_bad = np.zeros((8, 8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial dimensions"):
        optimize_graph_cut_change_map(d_u, d_u, guide_bad)

    # 1x1 pixel image
    d_u_1 = np.array([[2.0]], dtype=np.float32)
    d_c_1 = np.array([[0.5]], dtype=np.float32)
    guide_1 = np.zeros((1, 1, 4), dtype=np.float32)
    cmap_1, _ = optimize_graph_cut_change_map(d_u_1, d_c_1, guide_1)
    assert cmap_1.shape == (1, 1)
    assert cmap_1[0, 0] == 1

    cmap_1_fb, _ = optimize_graph_cut_change_map(d_u_1, d_c_1, guide_1, force_fallback=True)
    assert cmap_1_fb[0, 0] == 1

    # Empty array
    cmap_empty, _ = optimize_graph_cut_change_map(np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0, 4)))
    assert cmap_empty.size == 0


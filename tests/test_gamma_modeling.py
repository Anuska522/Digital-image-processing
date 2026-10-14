"""
Unit tests for Module 3: Localized Two-Parameter Gamma Statistical Modeling.
"""

import numpy as np
import pytest
from aggc_ssda.gamma_modeling import (
    compute_localized_distance_metric,
    fit_localized_gamma_distribution,
    compute_gamma_unary_costs,
)


def test_compute_localized_distance_metric():
    rng = np.random.RandomState(42)
    delta_f = rng.normal(0, 0.1, (30, 30, 4)).astype(np.float32)
    # Add strong change in center
    delta_f[10:20, 10:20, :] += 0.8

    y_map, meta = compute_localized_distance_metric(delta_f, window_size=7)
    assert y_map.shape == (30, 30)
    assert np.all(y_map > 0)
    # Changed region should have higher Y
    assert y_map[15, 15] > y_map[2, 2]


def test_fit_localized_gamma_distribution():
    rng = np.random.RandomState(42)
    # Generate Gamma-distributed numbers
    y_synth = rng.gamma(shape=3.0, scale=0.5, size=(40, 40)).astype(np.float32)
    y_synth = np.maximum(y_synth, 1e-4)

    k_map, theta_map, meta = fit_localized_gamma_distribution(y_synth, tile_size=20)
    assert k_map.shape == (40, 40)
    assert theta_map.shape == (40, 40)
    assert np.all(k_map > 0)
    assert np.all(theta_map > 0)
    # Should reasonably recover shape ~3 and scale ~0.5
    assert 1.0 <= meta["global_k"] <= 6.0
    assert 0.1 <= meta["global_theta"] <= 2.0


def test_compute_gamma_unary_costs():
    y_map = np.array([[0.1, 5.0]], dtype=np.float32)
    k_map = np.array([[2.0, 2.0]], dtype=np.float32)
    theta_map = np.array([[0.5, 0.5]], dtype=np.float32)

    d_unchanged, d_changed, prob = compute_gamma_unary_costs(y_map, k_map, theta_map)
    assert d_unchanged.shape == (1, 2)
    assert d_changed.shape == (1, 2)
    assert prob.shape == (1, 2)

    # For small Y (0.1), cost of unchanged should be lower than cost of changed
    assert d_unchanged[0, 0] <= d_changed[0, 0]
    # For large Y (5.0), cost of changed should be lower than cost of unchanged
    assert d_changed[0, 1] <= d_unchanged[0, 1]
    # Probability of change should be strictly increasing with Y
    assert prob[0, 1] > prob[0, 0]

"""
Unit tests for Module 5: Binary Change Map Generation and Evaluation.
"""

import numpy as np
import pytest
from aggc_ssda.evaluation import (
    generate_change_overlay,
    analyze_morphological_patches,
    evaluate_change_detection,
)


def test_generate_change_overlay():
    base = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    overlay = generate_change_overlay(base, mask)
    assert overlay.shape == (30, 30, 3)
    assert overlay.dtype == np.uint8
    # Overlay should have color inside mask
    assert np.any(overlay[15, 15] > 0)
    # Outside mask should be unchanged black
    assert np.all(overlay[2, 2] == 0)


def test_analyze_morphological_patches():
    mask = np.zeros((40, 40), dtype=np.uint8)
    # Two separate patches of size 4 (2x2) each
    mask[2:4, 2:4] = 1
    mask[10:12, 10:12] = 1

    stats = analyze_morphological_patches(mask)
    assert stats["patch_count"] == 2
    assert stats["min_patch_pixels"] == 4
    assert stats["max_patch_pixels"] == 4
    assert stats["mean_patch_pixels"] == 4.0


def test_evaluate_change_detection_without_ground_truth():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1

    res = evaluate_change_detection(mask, ground_truth_mask=None)
    assert res["has_ground_truth"] is False
    assert res["changed_pixels"] == 25
    assert res["total_pixels"] == 400
    assert "metrics" not in res  # Never fabricate metrics without ground truth!


def test_evaluate_change_detection_with_ground_truth():
    gt = np.zeros((20, 20), dtype=np.uint8)
    pred = np.zeros((20, 20), dtype=np.uint8)

    # 10 true change pixels
    gt[:10, :1] = 1
    # 8 TP, 2 FN, 2 FP
    pred[:8, :1] = 1
    pred[10:12, :1] = 1

    res = evaluate_change_detection(pred, ground_truth_mask=gt)
    assert res["has_ground_truth"] is True
    cm = res["confusion_matrix"]
    assert cm["true_positive"] == 8
    assert cm["false_positive"] == 2
    assert cm["false_negative"] == 2
    assert cm["true_negative"] == 400 - (8 + 2 + 2)

    metrics = res["metrics"]
    assert metrics["precision"] == 80.0
    assert metrics["recall"] == 80.0
    assert metrics["f1_score"] == 80.0
    assert metrics["commission_error"] == 20.0
    assert metrics["omission_error"] == 20.0
    assert metrics["overall_accuracy"] == 99.0


def test_generate_change_overlay_2_channel():
    # 2-channel base image should not crash and should produce 3-channel RGB overlay
    base_2c = np.ones((20, 20, 2), dtype=np.uint8) * 100
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    overlay = generate_change_overlay(base_2c, mask)
    assert overlay.shape == (20, 20, 3)
    assert overlay.dtype == np.uint8


def test_generate_change_overlay_uint16_and_float():
    # 16-bit high dynamic range input
    base_u16 = np.ones((20, 20, 3), dtype=np.uint16) * 10000
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    overlay = generate_change_overlay(base_u16, mask)
    assert overlay.shape == (20, 20, 3)
    assert overlay.dtype == np.uint8
    # Outside mask should be non-saturated
    assert overlay[0, 0, 0] < 255


def test_evaluate_change_detection_empty():
    with pytest.raises(ValueError, match="empty"):
        evaluate_change_detection(np.empty((0, 0), dtype=np.uint8))

    with pytest.raises(TypeError, match="NumPy ndarray"):
        evaluate_change_detection([0, 1, 0])


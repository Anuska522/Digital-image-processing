"""
Integration tests for the complete 5-module AGGC-SSDA pipeline.
"""

import os
import numpy as np
import pytest
from aggc_ssda.dataset_loader import generate_benchmark_satellite_pair
from aggc_ssda.pipeline import AGGCSSDAPipeline, PipelineResult


def test_full_pipeline_run():
    # Generate 128x128 benchmark pair for fast integration testing
    t1, t2, gt, meta = generate_benchmark_satellite_pair(height=128, width=128, seed=42)

    pipeline = AGGCSSDAPipeline(
        do_coregistration=True,
        do_radiometric_norm=True,
        rrn_method="pif",
        glcm_window=5,
        covariance_window=9,
        tile_size=32,
        inlier_percentile=85.0,
        alpha_significance=0.005,
        lambda_param=2.5,
        sigma_scale=1.0
    )

    result = pipeline.run(t1_img=t1, t2_img=t2, ground_truth_mask=gt)

    assert isinstance(result, PipelineResult)
    assert result.binary_change_map.shape == (128, 128)
    assert result.change_overlay.shape == (128, 128, 3)
    assert result.execution_time_seconds > 0.0

    # Intermediate arrays check
    assert result.feat_t1.shape == (128, 128, 4)
    assert result.feat_t2.shape == (128, 128, 4)
    assert result.distance_map_y.shape == (128, 128)
    assert result.k_map.shape == (128, 128)
    assert result.theta_map.shape == (128, 128)
    assert result.d_unchanged.shape == (128, 128)
    assert result.d_changed.shape == (128, 128)

    # Evaluation results check
    eval_res = result.evaluation_metrics
    assert eval_res["has_ground_truth"] is True
    metrics = eval_res["metrics"]
    assert "f1_score" in metrics
    assert "overall_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "commission_error" in metrics
    assert "omission_error" in metrics

    # The pipeline should detect real changes with high accuracy on the benchmark
    assert metrics["overall_accuracy"] > 90.0
    assert metrics["f1_score"] > 70.0

"""
Pipeline Orchestrator for AGGC-SSDA:
Adaptive Gamma-Graph Cut Spatial-Spectral Architecture

Coordinates the five sequential modules into a unified, reproducible workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import time

from .preprocessing import preprocess_multitemporal_pair
from .features import extract_spatial_spectral_features, compute_feature_difference
from .gamma_modeling import compute_localized_distance_metric, fit_localized_gamma_distribution, compute_gamma_unary_costs
from .graph_cut import optimize_graph_cut_change_map
from .evaluation import evaluate_change_detection, generate_change_overlay


@dataclass
class PipelineResult:
    """Stores full telemetry, intermediate arrays, and final outputs of the AGGC-SSDA pipeline."""
    # Input & Preprocessed
    t1_preprocessed: np.ndarray
    t2_preprocessed: np.ndarray
    
    # Feature Stacks & Differences
    feat_t1: np.ndarray
    feat_t2: np.ndarray
    delta_f: np.ndarray
    euclidean_diff: np.ndarray
    
    # Gamma Modeling
    distance_map_y: np.ndarray
    k_map: np.ndarray
    theta_map: np.ndarray
    d_unchanged: np.ndarray
    d_changed: np.ndarray
    prob_change: np.ndarray
    
    # Graph-Cut & Final
    binary_change_map: np.ndarray
    change_overlay: np.ndarray
    
    # Diagnostics & Metrics
    module_diagnostics: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


class AGGCSSDAPipeline:
    """
    End-to-End AGGC-SSDA Unsupervised Change Detection Runner.
    """

    def __init__(
        self,
        do_coregistration: bool = True,
        do_radiometric_norm: bool = True,
        rrn_method: str = "pif",
        glcm_window: int = 7,
        covariance_window: int = 15,
        tile_size: int = 64,
        inlier_percentile: float = 85.0,
        alpha_significance: float = 0.005,
        lambda_param: float = 2.0,
        sigma_scale: float = 1.0
    ):
        self.do_coregistration = do_coregistration
        self.do_radiometric_norm = do_radiometric_norm
        self.rrn_method = rrn_method
        self.glcm_window = glcm_window
        self.covariance_window = covariance_window
        self.tile_size = tile_size
        self.inlier_percentile = inlier_percentile
        self.alpha_significance = alpha_significance
        self.lambda_param = lambda_param
        self.sigma_scale = sigma_scale

    def run(
        self,
        t1_img: np.ndarray,
        t2_img: np.ndarray,
        ground_truth_mask: Optional[np.ndarray] = None,
        band_mapping: Optional[Dict[str, int]] = None
    ) -> PipelineResult:
        """
        Executes the full 5-module AGGC-SSDA change detection pipeline.
        
        Args:
            t1_img: Temporal baseline acquisition image.
            t2_img: Temporal post-event acquisition image.
            ground_truth_mask: Optional binary ground-truth change mask for validation.
            band_mapping: Optional band mapping dict for multispectral input.
            
        Returns:
            PipelineResult containing all intermediate and final outputs.
        """
        start_time = time.time()
        diagnostics: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # MODULE 1: Preprocessing & Radiometric Calibration
        # -------------------------------------------------------------
        t1_prep, t2_prep, m1_diag = preprocess_multitemporal_pair(
            t1_img=t1_img,
            t2_img=t2_img,
            do_coregistration=self.do_coregistration,
            do_radiometric_norm=self.do_radiometric_norm,
            rrn_method=self.rrn_method
        )
        diagnostics["module_1_preprocessing"] = m1_diag

        # -------------------------------------------------------------
        # MODULE 2: Spatial-Spectral Feature Extraction
        # -------------------------------------------------------------
        feat_t1, m2_t1_diag = extract_spatial_spectral_features(
            img=t1_prep,
            band_mapping=band_mapping,
            glcm_window=self.glcm_window
        )
        feat_t2, m2_t2_diag = extract_spatial_spectral_features(
            img=t2_prep,
            band_mapping=band_mapping,
            glcm_window=self.glcm_window
        )
        delta_f, euclidean_diff = compute_feature_difference(feat_t1, feat_t2)
        diagnostics["module_2_features"] = {
            "t1_metadata": m2_t1_diag,
            "t2_metadata": m2_t2_diag,
            "mean_euclidean_diff": float(euclidean_diff.mean())
        }

        # -------------------------------------------------------------
        # MODULE 3: Localized Two-Parameter Gamma Statistical Modeling
        # -------------------------------------------------------------
        y_map, m3_dist_diag = compute_localized_distance_metric(
            delta_f=delta_f,
            window_size=self.covariance_window
        )
        k_map, theta_map, m3_fit_diag = fit_localized_gamma_distribution(
            y_map=y_map,
            tile_size=self.tile_size,
            inlier_percentile=self.inlier_percentile
        )
        d_unchanged, d_changed, prob_change = compute_gamma_unary_costs(
            y_map=y_map,
            k_map=k_map,
            theta_map=theta_map,
            alpha_significance=self.alpha_significance
        )
        diagnostics["module_3_gamma"] = {
            "distance_metric": m3_dist_diag,
            "gamma_parameters": m3_fit_diag,
            "mean_prob_change": float(prob_change.mean())
        }

        # -------------------------------------------------------------
        # MODULE 4: Contrast-Weighted Graph-Cut Optimization
        # -------------------------------------------------------------
        binary_change_map, m4_diag = optimize_graph_cut_change_map(
            d_unchanged=d_unchanged,
            d_changed=d_changed,
            feature_guide=feat_t1,
            lambda_param=self.lambda_param,
            sigma_scale=self.sigma_scale
        )
        diagnostics["module_4_graph_cut"] = m4_diag

        # -------------------------------------------------------------
        # MODULE 5: Binary Change Map Generation & Evaluation
        # -------------------------------------------------------------
        overlay = generate_change_overlay(
            base_img=t2_prep,
            binary_change_map=binary_change_map
        )
        evaluation_results = evaluate_change_detection(
            predicted_change_map=binary_change_map,
            ground_truth_mask=ground_truth_mask
        )
        diagnostics["module_5_evaluation"] = evaluation_results

        total_time = time.time() - start_time

        return PipelineResult(
            t1_preprocessed=t1_prep,
            t2_preprocessed=t2_prep,
            feat_t1=feat_t1,
            feat_t2=feat_t2,
            delta_f=delta_f,
            euclidean_diff=euclidean_diff,
            distance_map_y=y_map,
            k_map=k_map,
            theta_map=theta_map,
            d_unchanged=d_unchanged,
            d_changed=d_changed,
            prob_change=prob_change,
            binary_change_map=binary_change_map,
            change_overlay=overlay,
            module_diagnostics=diagnostics,
            evaluation_metrics=evaluation_results,
            execution_time_seconds=round(total_time, 3)
        )

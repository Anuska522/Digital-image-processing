"""
AGGC-SSDA: Adaptive Gamma-Graph Cut Spatial-Spectral Architecture
for Unsupervised Earth Observation Mapping.

Implementation of the 5-module change detection methodology for multitemporal satellite imagery.
"""

from .preprocessing import preprocess_multitemporal_pair, subpixel_coregistration, relative_radiometric_normalization
from .features import extract_spatial_spectral_features, compute_feature_difference
from .gamma_modeling import compute_localized_distance_metric, fit_localized_gamma_distribution, compute_gamma_unary_costs
from .graph_cut import optimize_graph_cut_change_map
from .evaluation import evaluate_change_detection, generate_change_overlay
from .pipeline import AGGCSSDAPipeline, PipelineResult

__version__ = "1.0.0"
__author__ = "Anuska Saha"

__all__ = [
    "preprocess_multitemporal_pair",
    "subpixel_coregistration",
    "relative_radiometric_normalization",
    "extract_spatial_spectral_features",
    "compute_feature_difference",
    "compute_localized_distance_metric",
    "fit_localized_gamma_distribution",
    "compute_gamma_unary_costs",
    "optimize_graph_cut_change_map",
    "evaluate_change_detection",
    "generate_change_overlay",
    "AGGCSSDAPipeline",
    "PipelineResult",
]

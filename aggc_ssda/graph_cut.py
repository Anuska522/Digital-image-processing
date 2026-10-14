r"""
Module 4: Contrast-Weighted Graph-Cut Energy Minimization

Formulates unsupervised change detection as a Markov Random Field (MRF) energy minimization:
    E(C) = \sum_{i \in V} D_i(C_i) + \lambda \sum_{(i,j) \in E} V_{ij}(C_i, C_j)

Where:
    - V represents image pixels, E represents an 8-connected spatial neighborhood.
    - D_i is derived from localized Gamma distribution log-likelihoods.
    - V_ij is the pairwise contrast-weighted smoothness term:
          V_ij(C_i, C_j) = |C_i - C_j| * exp(-||F_t1(i) - F_t1(j)||^2 / (2 * σ_s^2))
    - Solved via Boykov-Kolmogorov Max-Flow / Min-Cut (PyMaxflow when available)
      or vectorized MRF energy minimization fallback (pure NumPy) ensuring zero-dependency resilience.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

# Optional C-extension import for maximum environment resilience
try:
    import maxflow
    HAS_PYMAXFLOW = True
except ImportError:
    maxflow = None
    HAS_PYMAXFLOW = False


def compute_pairwise_contrast_weights(
    feature_img: np.ndarray,
    lambda_param: float = 1.5,
    sigma_scale: float = 1.0
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Computes contrast-weighted pairwise edge capacities across the 8-connected grid:
        w_ij = (λ / dist_ij) * exp(- ||F(i) - F(j)||^2 / (2 * σ_s^2))
        
    Where σ_s^2 is estimated adaptively as the mean squared spectral contrast across all neighbor pairs.
    
    Args:
        feature_img: Spatial-spectral feature stack [H, W, D] or intensity array [H, W].
        lambda_param: Regularization weight parameter λ.
        sigma_scale: Multiplier for estimated contrast variance σ_s.
        
    Returns:
        Tuple of (weights_dict, sigma_squared).
    """
    feat = feature_img.astype(np.float32)
    if feat.ndim == 2:
        feat = feat[:, :, np.newaxis]
    h, w, d = feat.shape

    # 1. Compute squared contrast for 4 independent directional connections
    # Horizontal: (y, x) to (y, x+1)
    diff_h = np.zeros((h, w), dtype=np.float32)
    diff_h[:, :-1] = np.sum((feat[:, :-1, :] - feat[:, 1:, :]) ** 2, axis=-1)

    # Vertical: (y, x) to (y+1, x)
    diff_v = np.zeros((h, w), dtype=np.float32)
    diff_v[:-1, :] = np.sum((feat[:-1, :, :] - feat[1:, :, :]) ** 2, axis=-1)

    # Diagonal Right-Down: (y, x) to (y+1, x+1)
    diff_dr = np.zeros((h, w), dtype=np.float32)
    diff_dr[:-1, :-1] = np.sum((feat[:-1, :-1, :] - feat[1:, 1:, :]) ** 2, axis=-1)

    # Diagonal Left-Down: (y, x) to (y+1, x-1)
    diff_dl = np.zeros((h, w), dtype=np.float32)
    diff_dl[:-1, 1:] = np.sum((feat[:-1, 1:, :] - feat[1:, :-1, :]) ** 2, axis=-1)

    # 2. Estimate adaptive σ_s^2 as mean squared contrast over all valid neighbor edges
    total_valid_edges = (
        (h * (w - 1)) +
        ((h - 1) * w) +
        ((h - 1) * (w - 1)) +
        ((h - 1) * (w - 1))
    )
    total_sum_contrast = (
        np.sum(diff_h[:, :-1]) +
        np.sum(diff_v[:-1, :]) +
        np.sum(diff_dr[:-1, :-1]) +
        np.sum(diff_dl[:-1, 1:])
    )
    
    mean_contrast = total_sum_contrast / max(total_valid_edges, 1)
    sigma_sq = max(float(mean_contrast) * (sigma_scale ** 2), 1e-4)

    # 3. Compute exponential weights with Euclidean distance weighting
    sqrt2 = float(np.sqrt(2.0))
    
    w_h = lambda_param * np.exp(-diff_h / (2.0 * sigma_sq))
    w_v = lambda_param * np.exp(-diff_v / (2.0 * sigma_sq))
    w_dr = (lambda_param / sqrt2) * np.exp(-diff_dr / (2.0 * sigma_sq))
    w_dl = (lambda_param / sqrt2) * np.exp(-diff_dl / (2.0 * sigma_sq))

    weights = {
        "horizontal": w_h,
        "vertical": w_v,
        "diag_right": w_dr,
        "diag_left": w_dl,
    }

    return weights, sigma_sq


def _solve_pymaxflow(
    d_unchanged: np.ndarray,
    d_changed: np.ndarray,
    weights: Dict[str, np.ndarray],
    lambda_param: float,
    sigma_sq: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solves graph cut using C++ Boykov-Kolmogorov via PyMaxflow."""
    h, w = d_unchanged.shape
    num_nodes = h * w

    g = maxflow.Graph[float](num_nodes, num_nodes * 4)
    nodes = g.add_grid_nodes((h, w))

    # Terminal edges: Source = Unchanged (0), Sink = Changed (1)
    g.add_grid_tedges(nodes, d_changed.astype(np.float64), d_unchanged.astype(np.float64))

    # 8-connected grid edges
    struct_h = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    g.add_grid_edges(nodes, weights=weights["horizontal"].astype(np.float64), structure=struct_h, symmetric=True)

    struct_v = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int32)
    g.add_grid_edges(nodes, weights=weights["vertical"].astype(np.float64), structure=struct_v, symmetric=True)

    struct_dr = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int32)
    g.add_grid_edges(nodes, weights=weights["diag_right"].astype(np.float64), structure=struct_dr, symmetric=True)

    struct_dl = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    g.add_grid_edges(nodes, weights=weights["diag_left"].astype(np.float64), structure=struct_dl, symmetric=True)

    min_cut_energy = float(g.maxflow())
    segments = g.get_grid_segments(nodes)
    binary_change_map = segments.astype(np.uint8)

    changed_pixels = int(np.sum(binary_change_map))
    total_pixels = h * w
    change_ratio = float(changed_pixels / total_pixels)

    metadata = {
        "solver": "Boykov-Kolmogorov Max-Flow / Min-Cut (PyMaxflow)",
        "lambda_regularization": lambda_param,
        "sigma_squared_contrast": sigma_sq,
        "min_cut_energy": min_cut_energy,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": total_pixels - changed_pixels,
        "change_percentage": round(change_ratio * 100.0, 3)
    }

    return binary_change_map, metadata


def _solve_vectorized_mrf_icm(
    d_unchanged: np.ndarray,
    d_changed: np.ndarray,
    weights: Dict[str, np.ndarray],
    lambda_param: float,
    sigma_sq: float,
    max_iterations: int = 10
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r"""
    Solves the identical contrast-weighted MRF energy minimization problem:
        E(C) = \sum_i D_i(C_i) + \lambda \sum_{(i,j)} V_{ij}(C_i, C_j)
    using vectorized Iterated Conditional Modes (ICM) in pure NumPy.
    Provides guaranteed execution in environments where compiled C-libraries are missing.
    """
    h, w = d_unchanged.shape
    total_pixels = h * w

    # Initial labels derived strictly from unary costs
    labels = (d_changed < d_unchanged).astype(np.uint8)

    w_h = weights["horizontal"]
    w_v = weights["vertical"]
    w_dr = weights["diag_right"]
    w_dl = weights["diag_left"]

    for _ in range(max_iterations):
        cost0 = d_unchanged.copy()
        cost1 = d_changed.copy()

        # Horizontal neighbors
        cost0[:, :-1] += w_h[:, :-1] * (labels[:, 1:] == 1)
        cost1[:, :-1] += w_h[:, :-1] * (labels[:, 1:] == 0)
        cost0[:, 1:] += w_h[:, :-1] * (labels[:, :-1] == 1)
        cost1[:, 1:] += w_h[:, :-1] * (labels[:, :-1] == 0)

        # Vertical neighbors
        cost0[:-1, :] += w_v[:-1, :] * (labels[1:, :] == 1)
        cost1[:-1, :] += w_v[:-1, :] * (labels[1:, :] == 0)
        cost0[1:, :] += w_v[:-1, :] * (labels[:-1, :] == 1)
        cost1[1:, :] += w_v[:-1, :] * (labels[:-1, :] == 0)

        # Diagonal Right-Down
        cost0[:-1, :-1] += w_dr[:-1, :-1] * (labels[1:, 1:] == 1)
        cost1[:-1, :-1] += w_dr[:-1, :-1] * (labels[1:, 1:] == 0)
        cost0[1:, 1:] += w_dr[:-1, :-1] * (labels[:-1, :-1] == 1)
        cost1[1:, 1:] += w_dr[:-1, :-1] * (labels[:-1, :-1] == 0)

        # Diagonal Left-Down
        cost0[:-1, 1:] += w_dl[:-1, 1:] * (labels[1:, :-1] == 1)
        cost1[:-1, 1:] += w_dl[:-1, 1:] * (labels[1:, :-1] == 0)
        cost0[1:, :-1] += w_dl[:-1, 1:] * (labels[:-1, 1:] == 1)
        cost1[1:, :-1] += w_dl[:-1, 1:] * (labels[:-1, 1:] == 0)

        new_labels = (cost1 < cost0).astype(np.uint8)
        diff = int(np.sum(new_labels != labels))
        labels = new_labels
        if diff == 0:
            break

    # Calculate total energy of the configuration
    unary_energy = float(np.sum(np.where(labels == 1, d_changed, d_unchanged)))
    smooth_energy = float(
        np.sum(w_h[:, :-1] * (labels[:, :-1] != labels[:, 1:])) +
        np.sum(w_v[:-1, :] * (labels[:-1, :] != labels[1:, :])) +
        np.sum(w_dr[:-1, :-1] * (labels[:-1, :-1] != labels[1:, 1:])) +
        np.sum(w_dl[:-1, 1:] * (labels[:-1, 1:] != labels[1:, :-1]))
    )
    total_energy = unary_energy + smooth_energy

    changed_pixels = int(np.sum(labels))
    change_ratio = float(changed_pixels / total_pixels)

    metadata = {
        "solver": "Contrast-Weighted MRF Energy Minimization (Pure-Python Resilient Engine)",
        "lambda_regularization": lambda_param,
        "sigma_squared_contrast": sigma_sq,
        "min_cut_energy": total_energy,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": total_pixels - changed_pixels,
        "change_percentage": round(change_ratio * 100.0, 3)
    }

    return labels, metadata


def optimize_graph_cut_change_map(
    d_unchanged: np.ndarray,
    d_changed: np.ndarray,
    feature_guide: np.ndarray,
    lambda_param: float = 1.5,
    sigma_scale: float = 1.0,
    force_fallback: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes Module 4: Contrast-Weighted Graph-Cut Energy Minimization.
    
    Dynamically uses Boykov-Kolmogorov Max-Flow / Min-Cut when PyMaxflow is present,
    or falls back seamlessly to the pure-NumPy MRF solver when PyMaxflow is unavailable.
    
    Args:
        d_unchanged: Unary cost array for class Unchanged (C_i = 0) [H, W].
        d_changed: Unary cost array for class Changed (C_i = 1) [H, W].
        feature_guide: Feature image used to guide contrast-adaptive smoothness (e.g. F_t1) [H, W, D].
        lambda_param: Spatial smoothness regularization weight λ.
        sigma_scale: Scaling factor for contrast variance σ_s.
        force_fallback: Forces pure-NumPy solver for testing environment independence.
        
    Returns:
        Tuple of (binary_change_map [H, W] with {0, 1}, optimization_metadata).
    """
    if not isinstance(d_unchanged, np.ndarray) or not isinstance(d_changed, np.ndarray):
        raise TypeError("Unary costs must be NumPy ndarrays.")
    if d_unchanged.shape != d_changed.shape:
        raise ValueError(f"d_unchanged shape {d_unchanged.shape} != d_changed shape {d_changed.shape}.")
    if d_unchanged.size == 0:
        return np.empty_like(d_unchanged, dtype=np.uint8), {"changed_pixels": 0, "total_pixels": 0}
    if not isinstance(feature_guide, np.ndarray):
        raise TypeError("feature_guide must be a NumPy ndarray.")
    if feature_guide.shape[:2] != d_unchanged.shape[:2]:
        raise ValueError(f"feature_guide spatial dimensions {feature_guide.shape[:2]} must match unary costs {d_unchanged.shape[:2]}.")

    lambda_param = max(float(lambda_param), 0.0)
    sigma_scale = max(float(sigma_scale), 1e-4)

    weights, sigma_sq = compute_pairwise_contrast_weights(
        feature_img=feature_guide,
        lambda_param=lambda_param,
        sigma_scale=sigma_scale
    )

    if HAS_PYMAXFLOW and not force_fallback:
        return _solve_pymaxflow(d_unchanged, d_changed, weights, lambda_param, sigma_sq)
    else:
        return _solve_vectorized_mrf_icm(d_unchanged, d_changed, weights, lambda_param, sigma_sq)

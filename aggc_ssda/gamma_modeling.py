r"""
Module 3: Localized Two-Parameter Gamma Statistical Distribution Modeling

Implements:
1. Localized covariance-weighted distance metric:
       Y(x, y) = ΔF(x, y)^T Σ_local^{-1} ΔF(x, y)
   calculated over spatial sliding windows / tiled sub-regions using inlier background covariance.
2. Localized two-parameter Gamma distribution parameter estimation (shape k, scale θ)
   for the unchanged background class using Maximum Likelihood / Method of Moments.
3. Unary graph energy data costs:
       D_i(C_i = 0) = -ln(1 - \int_0^{Y_i} f(t; k_i, \theta_i) dt) = -ln(Survival(Y_i; k_i, \theta_i))
       D_i(C_i = 1) = -ln(\alpha)
   where \alpha is the significance threshold for change isolation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter


def compute_localized_distance_metric(
    delta_f: np.ndarray,
    window_size: int = 15,
    inlier_percentile: float = 85.0,
    regularization_eps: float = 1e-4
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computes localized covariance-weighted distance scalar Y(x, y):
        Y(x, y) = ΔF(x, y)^T Σ_local^{-1} ΔF(x, y)
        
    Estimates the localized background covariance Σ_local over spatial sliding windows / tiles
    from inlier difference pixels to prevent true changes from corrupting the background covariance.
    
    Args:
        delta_f: Difference feature stack [H, W, D].
        window_size: Neighborhood window size N for local covariance.
        inlier_percentile: Upper percentile of difference magnitude considered background.
        regularization_eps: Diagonal Tikhonov stabilizer.
        
    Returns:
        Tuple of (Y [H, W], metadata).
    """
    if not isinstance(delta_f, np.ndarray):
        raise TypeError("delta_f must be a NumPy ndarray.")
    if delta_f.size == 0:
        raise ValueError("delta_f cannot be empty.")
    if delta_f.ndim == 2:
        delta_f = delta_f[:, :, np.newaxis]
    elif delta_f.ndim != 3:
        raise ValueError(f"delta_f must be 2D or 3D, got ndim={delta_f.ndim}")

    h, w, d = delta_f.shape

    # 1. Magnitude of differences to isolate stable background pixels
    mag = np.linalg.norm(delta_f, axis=-1)
    inlier_threshold = float(np.percentile(mag, inlier_percentile))
    inlier_mask = mag <= inlier_threshold

    # Fallback if too few inliers
    if np.sum(inlier_mask) < 50:
        inlier_mask = np.ones((h, w), dtype=bool)

    # 2. Estimate localized background covariance across spatial tiles / windows
    # For robust inversion and smooth spatial transitions, compute tiled inlier covariance
    tile_h = max(window_size * 2, 32)
    tile_w = max(window_size * 2, 32)

    # Global background covariance as safe prior
    diff_inliers_global = delta_f[inlier_mask]
    if len(diff_inliers_global) > d:
        cov_global = np.cov(diff_inliers_global.T) + regularization_eps * np.eye(d)
    else:
        cov_global = np.eye(d, dtype=np.float32) * 0.05
    inv_cov_global = np.linalg.inv(cov_global)

    y_map = np.zeros((h, w), dtype=np.float32)

    num_tiles_y = int(np.ceil(h / tile_h))
    num_tiles_x = int(np.ceil(w / tile_w))

    for ty in range(num_tiles_y):
        y_start = ty * tile_h
        y_end = min(y_start + tile_h, h)
        for tx in range(num_tiles_x):
            x_start = tx * tile_w
            x_end = min(x_start + tile_w, w)

            tile_df = delta_f[y_start:y_end, x_start:x_end]
            tile_mask = inlier_mask[y_start:y_end, x_start:x_end]

            pts = tile_df[tile_mask]
            if len(pts) > d * 4:
                cov_local = np.cov(pts.T) + regularization_eps * np.eye(d)
                try:
                    inv_cov_local = np.linalg.inv(cov_local)
                except np.linalg.LinAlgError:
                    inv_cov_local = inv_cov_global
            else:
                inv_cov_local = inv_cov_global

            # Compute quadratic form: ΔF^T Σ_local^{-1} ΔF
            tile_flat = tile_df.reshape(-1, d)
            y_tile = np.sum((tile_flat @ inv_cov_local) * tile_flat, axis=-1)
            y_map[y_start:y_end, x_start:x_end] = y_tile.reshape(y_end - y_start, x_end - x_start)

    # Enforce strictly positive values
    y_map = np.maximum(y_map, 1e-5).astype(np.float32)

    metadata = {
        "metric_type": "Localized Covariance-Weighted Distance Y",
        "window_size": window_size,
        "mean_Y": float(y_map.mean()),
        "median_Y": float(np.median(y_map)),
        "max_Y": float(y_map.max()),
        "min_Y": float(y_map.min()),
        "inlier_threshold": inlier_threshold
    }

    return y_map, metadata


def fit_localized_gamma_distribution(
    y_map: np.ndarray,
    tile_size: int = 64,
    inlier_percentile: float = 85.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits localized two-parameter Gamma distribution f(Y; k, θ) across spatial tiles.
    
    The unchanged background class conforms to the Gamma distribution. Inlier filtering
    (retaining lower tail below inlier_percentile) prevents changed pixels from distorting
    the estimated background null distribution.
    
    Args:
        y_map: Continuous distance scalar map Y [H, W].
        tile_size: Spatial tile dimension for localized parameter adaptation.
        inlier_percentile: Cutoff percentile to isolate stable background pixels.
        
    Returns:
        Tuple of (k_map [H, W], theta_map [H, W], metadata).
    """
    h, w = y_map.shape
    k_map = np.zeros((h, w), dtype=np.float32)
    theta_map = np.zeros((h, w), dtype=np.float32)

    # Estimate global baseline first as safe fallback
    y_flat = y_map.ravel()
    thresh_global = float(np.percentile(y_flat, inlier_percentile))
    inliers_global = y_flat[y_flat <= thresh_global]
    inliers_global = inliers_global[inliers_global > 1e-6]

    if len(inliers_global) > 1 and np.var(inliers_global) > 1e-8:
        mean_g = float(np.mean(inliers_global))
        var_g = float(np.var(inliers_global))
        if var_g > 1e-8 and mean_g > 1e-6:
            k_global = np.clip(mean_g**2 / var_g, 0.2, 50.0)
            theta_global = np.clip(var_g / mean_g, 1e-3, 100.0)
        else:
            k_global = 2.0
            theta_global = 1.0
    else:
        k_global = 2.0
        theta_global = 1.0

    # Local tile-based estimation
    step = max(int(tile_size), 4)
    num_tiles_y = int(np.ceil(h / step))
    num_tiles_x = int(np.ceil(w / step))

    for ty in range(num_tiles_y):
        y_start = ty * step
        y_end = min(y_start + step, h)
        for tx in range(num_tiles_x):
            x_start = tx * step
            x_end = min(x_start + step, w)

            tile_y = y_map[y_start:y_end, x_start:x_end].ravel()
            thresh_tile = float(np.percentile(tile_y, inlier_percentile))
            inliers_tile = tile_y[tile_y <= thresh_tile]
            inliers_tile = inliers_tile[inliers_tile > 1e-6]

            if len(inliers_tile) > 20 and np.var(inliers_tile) > 1e-8:
                m_t = float(np.mean(inliers_tile))
                v_t = float(np.var(inliers_tile))
                k_val = np.clip(m_t**2 / v_t, 0.2, 50.0)
                theta_val = np.clip(v_t / m_t, 1e-3, 100.0)
            else:
                k_val = k_global
                theta_val = theta_global

            k_map[y_start:y_end, x_start:x_end] = k_val
            theta_map[y_start:y_end, x_start:x_end] = theta_val

    # Smooth parameter maps across tile boundaries to eliminate artificial seams
    k_map = uniform_filter(k_map, size=max(step // 2, 3), mode='reflect')
    theta_map = uniform_filter(theta_map, size=max(step // 2, 3), mode='reflect')

    metadata = {
        "global_k": float(k_global),
        "global_theta": float(theta_global),
        "mean_local_k": float(k_map.mean()),
        "mean_local_theta": float(theta_map.mean()),
        "tile_size": step,
        "inlier_percentile": inlier_percentile
    }

    return k_map, theta_map, metadata


def compute_gamma_unary_costs(
    y_map: np.ndarray,
    k_map: np.ndarray,
    theta_map: np.ndarray,
    alpha_significance: float = 0.01,
    clip_max_cost: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Computes unary data costs for graph cut optimization based on the local Gamma distribution:
        D_i(C_i = 0) = -ln(1 - \int_0^{Y_i} f(t; k_i, \theta_i) dt) = -ln(Survival_Function(Y_i))
        D_i(C_i = 1) = -ln(\alpha)
        
    Where \alpha is the critical significance threshold (e.g. 0.01) isolating upper-tail changes.
    When Y_i is typical background, Survival(Y_i) is large, so D_i(0) is low.
    When Y_i is in the extreme tail, Survival(Y_i) < \alpha, so D_i(0) exceeds D_i(1),
    favoring the Change label C_i = 1.
    
    Args:
        y_map: Difference metric scalar map Y [H, W].
        k_map: Local shape parameters k [H, W].
        theta_map: Local scale parameters θ [H, W].
        alpha_significance: Statistical upper-tail significance threshold.
        clip_max_cost: Maximum cost ceiling to prevent infinite unary energies.
        
    Returns:
        Tuple of (D_unchanged [H, W], D_changed [H, W], probability_change [H, W]).
    """
    y_safe = np.maximum(y_map, 1e-5)
    k_safe = np.maximum(k_map, 0.1)
    theta_safe = np.maximum(theta_map, 1e-4)

    # 1. Survival function of unchanged background: S(Y) = 1 - F(Y)
    # Using scipy.stats.gamma.logsf for high precision without underflow
    log_sf = stats.gamma.logsf(y_safe, a=k_safe, scale=theta_safe)
    # Cost for assigning Unchanged (label 0):
    d_unchanged = -log_sf

    # 2. Cost for assigning Changed (label 1):
    # Safely bound alpha to (1e-12, 1.0 - 1e-12) to avoid NaN/Inf
    alpha_safe = float(np.clip(alpha_significance, 1e-12, 1.0 - 1e-12))
    d_changed = -np.log(alpha_safe) * np.ones_like(d_unchanged)

    # 3. Continuous change probability for visualization: P_change = 1 - survival
    prob_change = 1.0 - np.exp(np.clip(log_sf, -30.0, 0.0))
    prob_change = np.clip(prob_change, 0.0, 1.0).astype(np.float32)

    # Regularize and normalize unary costs
    d_unchanged = np.clip(d_unchanged, 0.0, clip_max_cost).astype(np.float32)
    d_changed = np.clip(d_changed, 0.0, clip_max_cost).astype(np.float32)

    return d_unchanged, d_changed, prob_change

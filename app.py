"""
AGGC-SSDA Interactive Earth Observation Change Detection Dashboard

Production-hardened Streamlit application for:
Land-Cover Change Detection using Adaptive Gamma-Graph Cut Spatial-Spectral Architecture (AGGC-SSDA)
for Unsupervised Earth Observation Mapping.
"""

from typing import Optional, Tuple, Dict, Any
import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

from aggc_ssda.dataset_loader import (
    generate_benchmark_satellite_pair,
    load_image_from_file,
)
from aggc_ssda.pipeline import AGGCSSDAPipeline, PipelineResult
from aggc_ssda.evaluation import generate_change_overlay, evaluate_change_detection
from aggc_ssda.security import safe_decode_image_bytes, SecurityValidationError


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AGGC-SSDA | Land-Cover Change Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #09131f 0%, #112233 50%, #162b44 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-title {
        color: #f1f5f9;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 6px;
        font-weight: 400;
        letter-spacing: 0.2px;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
        margin-right: 6px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin: 4px 0;
        letter-spacing: -0.5px;
    }

    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.6px;
    }

    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div style="margin-bottom: 6px;">
            <span class="status-badge">[SYSTEM ACTIVE]</span>
            <span class="status-badge">[UNSUPERVISED ARCHITECTURE]</span>
        </div>
        <h1 class="main-title">AGGC-SSDA: Unsupervised Earth Observation Change Detection</h1>
        <p class="main-subtitle">
            Adaptive Gamma-Graph Cut Spatial-Spectral Architecture | Digital Image Processing (BCSE403L) Review I
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def decode_uploaded_image(uploaded_file) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Safely decodes uploaded file bytes with multispectral channel preservation and resource bounds."""
    if uploaded_file is None:
        return None
    try:
        image_bytes = uploaded_file.read()
        arr, metadata = safe_decode_image_bytes(image_bytes, uploaded_file.name)
        return arr, metadata
    except SecurityValidationError as sve:
        st.error(f"Input Validation Error: {sve}")
        return None
    except Exception as e:
        st.error(f"Failed to read image {uploaded_file.name}: {e}")
        return None


def _prepare_for_display(img: np.ndarray) -> np.ndarray:
    """Prepares an image array for Streamlit st.image rendering, handling multispectral channels."""
    if img.ndim == 3 and img.shape[2] > 3:
        # Render the first 3 channels as standard RGB display
        norm = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        norm = img[:, :, 0]
    else:
        norm = img

    if norm.dtype != np.uint8:
        if norm.max() <= 1.0:
            return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            return np.clip(norm, 0, 255).astype(np.uint8)
    return norm.copy()


def array_to_png_bytes(img: np.ndarray) -> bytes:
    """Converts a NumPy image or mask array to PNG bytes for download."""
    if img.ndim == 3 and img.shape[2] > 3:
        to_save = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        to_save = img[:, :, 0]
    else:
        to_save = img

    if to_save.dtype != np.uint8:
        if to_save.max() <= 1.0:
            u8 = (np.clip(to_save, 0, 1) * 255).astype(np.uint8)
        else:
            u8 = np.clip(to_save, 0, 255).astype(np.uint8)
    else:
        u8 = to_save.copy()

    pil_img = Image.fromarray(u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()



# -----------------------------------------------------------------------------
# Sidebar: Configuration & Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Pipeline Configuration")

data_source = st.sidebar.radio(
    "Select Input Source:",
    options=["Benchmark Sample Pair (Deforestation & Urban Growth)", "Upload Custom Satellite Images"],
    index=0,
)

t1_input = None
t2_input = None
gt_input = None
t1_meta: Dict[str, Any] = {}
t2_meta: Dict[str, Any] = {}

if data_source == "Benchmark Sample Pair (Deforestation & Urban Growth)":
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample")
    t1_path = os.path.join(sample_dir, "t1_pre.png")
    t2_path = os.path.join(sample_dir, "t2_post.png")
    gt_path = os.path.join(sample_dir, "ground_truth.png")

    if os.path.exists(t1_path) and os.path.exists(t2_path):
        t1_input = load_image_from_file(t1_path)
        t2_input = load_image_from_file(t2_path)
        if os.path.exists(gt_path):
            gt_raw = load_image_from_file(gt_path)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[:, :, 0]
            gt_input = (gt_raw > 127).astype(np.uint8)
    else:
        t1_input, t2_input, gt_input, _ = generate_benchmark_satellite_pair(256, 256)

    st.sidebar.info("Using documented 256x256 Earth Observation pair with reference ground truth.")

else:
    st.sidebar.subheader("Upload Multitemporal Acquisitions")
    uploaded_t1 = st.sidebar.file_uploader(
        "1. Temporal Image T1 (Pre-event Reference)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Accepts optical RGB or multi-band GeoTIFF rasters (e.g. 4-band B, G, R, NIR)."
    )
    uploaded_t2 = st.sidebar.file_uploader(
        "2. Temporal Image T2 (Post-event Subject)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Accepts matching acquisition from subsequent date."
    )
    uploaded_gt = st.sidebar.file_uploader(
        "3. Optional Ground-Truth Mask (Validation)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Binary reference mask: 0 = Unchanged, 255/1 = Changed."
    )

    if uploaded_t1:
        res_t1 = decode_uploaded_image(uploaded_t1)
        if res_t1 is not None:
            t1_input, t1_meta = res_t1
    if uploaded_t2:
        res_t2 = decode_uploaded_image(uploaded_t2)
        if res_t2 is not None:
            t2_input, t2_meta = res_t2
    if uploaded_gt:
        res_gt = decode_uploaded_image(uploaded_gt)
        if res_gt is not None:
            gt_arr, _ = res_gt
            if gt_arr.ndim == 3:
                gt_arr = gt_arr[:, :, 0]
            gt_input = (gt_arr > (127 if gt_arr.max() > 1 else 0.5)).astype(np.uint8)


# Sidebar Parameters
with st.sidebar.expander("Module 1: Preprocessing & Radiometric Calibration", expanded=False):
    do_coreg = st.checkbox("Sub-Pixel Co-Registration", value=True, help="Aligns T2 to T1 via sub-pixel ECC / feature warping.")
    do_rrn = st.checkbox("Relative Radiometric Normalization (RRN)", value=True, help="Normalizes illumination via Pseudo-Invariant Features (PIFs).")
    rrn_method = st.selectbox("RRN Method", options=["pif", "histogram_matching"], index=0)

with st.sidebar.expander("Module 2: Feature Extraction", expanded=False):
    glcm_window = st.slider("GLCM Texture Window Size", min_value=3, max_value=15, value=7, step=2, help="Neighborhood size for GLCM structural entropy.")

with st.sidebar.expander("Module 3: Gamma Statistical Modeling", expanded=True):
    cov_window = st.slider("Covariance Window Size (N)", min_value=5, max_value=31, value=15, step=2, help="Window size for localized background covariance estimation.")
    tile_size = st.select_slider("Spatial Tile Size (px)", options=[16, 32, 64, 128], value=64, help="Block size for localized Gamma parameter fitting.")
    alpha_sig = st.slider("Significance Cutoff (alpha)", min_value=0.001, max_value=0.05, value=0.005, step=0.001, format="%.3f", help="Statistical p-value cutoff isolating upper-tail changes.")
    inlier_percentile = st.slider("Background Inlier Quantile (%)", min_value=70.0, max_value=95.0, value=85.0, step=1.0)

with st.sidebar.expander("Module 4: Graph-Cut Energy Minimization", expanded=True):
    lambda_param = st.slider("Smoothness Regularization (lambda)", min_value=0.1, max_value=5.0, value=2.0, step=0.1, help="Penalizes isolated salt-and-pepper noise while preserving edges.")
    sigma_scale = st.slider("Contrast Variance Scale (sigma_s)", min_value=0.2, max_value=2.0, value=1.0, step=0.1, help="Scales pairwise contrast edge relaxation.")

with st.sidebar.expander("Visualization & Overlay Styling", expanded=False):
    overlay_hex = st.color_picker("Change Overlay Color", value="#FF2D55")
    overlay_rgb = tuple(int(overlay_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    overlay_alpha = st.slider("Overlay Transparency (Alpha)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    draw_contours = st.checkbox("Draw Boundary Contours", value=True)


# -----------------------------------------------------------------------------
# Main Dashboard Body
# -----------------------------------------------------------------------------
if t1_input is None or t2_input is None:
    st.warning("Please provide both Temporal Image T1 and T2 using the sidebar to execute change detection.")
    st.stop()

# Execution Trigger
run_pipeline_button = st.button("Run AGGC-SSDA Change Detection Pipeline", type="primary")

# Compute configuration hash for efficient session state management
config_key = (
    id(t1_input), id(t2_input),
    do_coreg, do_rrn, rrn_method, glcm_window,
    cov_window, tile_size, alpha_sig, inlier_percentile,
    lambda_param, sigma_scale,
    gt_input is not None
)

if run_pipeline_button or "pipeline_result" not in st.session_state or st.session_state.get("pipeline_config_key") != config_key:
    with st.spinner("Executing 5-Module AGGC-SSDA Pipeline..."):
        pipeline = AGGCSSDAPipeline(
            do_coregistration=do_coreg,
            do_radiometric_norm=do_rrn,
            rrn_method=rrn_method,
            glcm_window=glcm_window,
            covariance_window=cov_window,
            tile_size=tile_size,
            inlier_percentile=inlier_percentile,
            alpha_significance=alpha_sig,
            lambda_param=lambda_param,
            sigma_scale=sigma_scale,
        )

        t_start = time.time()
        result = pipeline.run(
            t1_img=t1_input,
            t2_img=t2_input,
            ground_truth_mask=gt_input,
        )
        st.session_state["pipeline_result"] = result
        st.session_state["pipeline_config_key"] = config_key
        st.session_state["run_time"] = time.time() - t_start

res: PipelineResult = st.session_state["pipeline_result"]


# Dynamic Custom Overlay based on UI settings
current_overlay = generate_change_overlay(
    base_img=res.t2_preprocessed,
    binary_change_map=res.binary_change_map,
    overlay_color=overlay_rgb,
    alpha=overlay_alpha,
    draw_contours=draw_contours,
)


# -----------------------------------------------------------------------------
# Summary Metric Banners
# -----------------------------------------------------------------------------
eval_dict = res.evaluation_metrics
morph_dict = eval_dict.get("morphology", {})

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Changed Area</div>
            <div class="metric-value">{eval_dict.get('change_percentage', 0)}%</div>
            <div style="font-size: 11px; color: #94a3b8;">{eval_dict.get('changed_pixels', 0):,} pixels</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Change Patches</div>
            <div class="metric-value">{morph_dict.get('patch_count', 0)}</div>
            <div style="font-size: 11px; color: #94a3b8;">Mean: {morph_dict.get('mean_patch_pixels', 0):.0f} px</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    if eval_dict.get("has_ground_truth"):
        oa_val = eval_dict["metrics"]["overall_accuracy"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value" style="color: #4ade80;">{oa_val}%</div>
                <div style="font-size: 11px; color: #94a3b8;">Reference Validated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Validation Mode</div>
                <div class="metric-value" style="color: #facc15; font-size: 18px;">Unsupervised</div>
                <div style="font-size: 11px; color: #94a3b8;">No Reference Provided</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_m4:
    if eval_dict.get("has_ground_truth"):
        f1_val = eval_dict["metrics"]["f1_score"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: #38bdf8;">{f1_val}%</div>
                <div style="font-size: 11px; color: #94a3b8;">Precision: {eval_dict['metrics']['precision']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Graph Energy</div>
                <div class="metric-value" style="font-size: 18px;">{res.module_diagnostics['module_4_graph_cut']['min_cut_energy']:.1f}</div>
                <div style="font-size: 11px; color: #94a3b8;">Min-Cut / Max-Flow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_m5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Pipeline Runtime</div>
            <div class="metric-value" style="color: #c084fc;">{res.execution_time_seconds}s</div>
            <div style="font-size: 11px; color: #94a3b8;">5 Modules End-to-End</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# -----------------------------------------------------------------------------
# Main Visualization Tabs
# -----------------------------------------------------------------------------
view_tabs = st.tabs([
    "Final Results and Overlay",
    "Multitemporal Acquisitions",
    "Intermediate Feature Maps",
    "Gamma Statistics and Unary Costs",
    "Quantitative Evaluation and Diagnostics"
])


# TAB 1: Final Results
with view_tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Binary Change Map C(x, y)")
        st.caption("0 = Unchanged Background (Black) | 1 = Delineated Land-Cover Change (White)")
        st.image((res.binary_change_map * 255).astype(np.uint8))

    with col2:
        st.subheader("High-Visibility Translucent Change Overlay")
        st.caption("Optimal min-cut change boundary dynamically projected over post-event scene T2")
        st.image(current_overlay)

    # Downloads
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.download_button(
            "Download Change Mask (PNG)",
            data=array_to_png_bytes(res.binary_change_map * 255),
            file_name="aggc_ssda_change_map.png",
            mime="image/png"
        )
    with d_col2:
        st.download_button(
            "Download Change Overlay (PNG)",
            data=array_to_png_bytes(current_overlay),
            file_name="aggc_ssda_overlay.png",
            mime="image/png"
        )
    with d_col3:
        report_json = json.dumps(res.module_diagnostics, indent=2)
        st.download_button(
            "Download Diagnostics (JSON)",
            data=report_json,
            file_name="aggc_ssda_diagnostics.json",
            mime="application/json"
        )
    with d_col4:
        summary_df = pd.DataFrame([
            {"Metric": "Changed Pixels", "Value": eval_dict.get("changed_pixels")},
            {"Metric": "Total Pixels", "Value": eval_dict.get("total_pixels")},
            {"Metric": "Change Ratio (%)", "Value": eval_dict.get("change_percentage")},
            {"Metric": "Patch Count", "Value": morph_dict.get("patch_count")},
            {"Metric": "Overall Accuracy (%)", "Value": eval_dict.get("metrics", {}).get("overall_accuracy", "N/A")},
            {"Metric": "F1-Score (%)", "Value": eval_dict.get("metrics", {}).get("f1_score", "N/A")},
            {"Metric": "Execution Time (s)", "Value": res.execution_time_seconds},
        ])
        st.download_button(
            "Download Summary (CSV)",
            data=summary_df.to_csv(index=False),
            file_name="aggc_ssda_summary.csv",
            mime="text/csv"
        )


# TAB 2: Multitemporal Acquisitions
with view_tabs[1]:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Temporal Acquisition T1 (Pre-event)")
        st.caption("Reference image baseline after radiometric scaling")
        st.image(_prepare_for_display(res.t1_preprocessed))

    with col_t2:
        st.subheader("Temporal Acquisition T2 (Post-event)")
        st.caption("Subject image following sub-pixel co-registration and PIF normalization")
        st.image(_prepare_for_display(res.t2_preprocessed))


    with st.expander("Module 1 Preprocessing Telemetry", expanded=True):
        m1_meta = res.module_diagnostics.get("module_1_preprocessing", {})
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Spatial Co-Registration:**")
            st.json(m1_meta.get("coregistration", {}))
        with c2:
            st.write("**Radiometric Normalization (RRN):**")
            st.json(m1_meta.get("radiometric_normalization", {}))


# TAB 3: Intermediate Feature Maps
with view_tabs[2]:
    st.markdown("### 4-Dimensional Spatial-Spectral Feature Components")
    f_meta_t1 = res.module_diagnostics.get("module_2_features", {}).get("t1_metadata", {})
    f_mode = f_meta_t1.get("mode", "Standard")
    st.info(f"Active Feature Extraction Mode: {f_mode}")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.write("**Visible Red Reflectance (R_vis)**")
        st.image(res.feat_t1[:, :, 0], clamp=True)
    with f_col2:
        st.write("**Near-Infrared / Canopy Density**")
        st.image(res.feat_t1[:, :, 1], clamp=True)
    with f_col3:
        st.write("**Tasseled Cap / Moisture**")
        st.image(res.feat_t1[:, :, 2], clamp=True)
    with f_col4:
        st.write("**GLCM Structural Entropy**")
        st.image(res.feat_t1[:, :, 3], clamp=True)

    st.write("---")
    st.markdown("### Multitemporal Difference Representations")
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Euclidean Difference Magnitude ||Delta F||_2")
        st.caption("Direct unweighted Euclidean spectral distance")
        st.image(res.euclidean_diff / (res.euclidean_diff.max() + 1e-6), clamp=True)
    with d2:
        st.subheader("Continuous Distance Scalar Y(x, y)")
        st.caption("Localized covariance-weighted distance metric: Delta F^T Sigma_local^{-1} Delta F")
        st.image(res.distance_map_y / (res.distance_map_y.max() + 1e-6), clamp=True)


# TAB 4: Gamma Statistics & Unary Costs
with view_tabs[3]:
    st.markdown("### Localized Gamma Distribution Fitting & Energy Modeling")
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.write("**Fitted Shape Parameter (k)**")
        st.caption(f"Mean: {res.k_map.mean():.2f} | Min: {res.k_map.min():.2f} | Max: {res.k_map.max():.2f}")
        st.image(res.k_map / (res.k_map.max() + 1e-6), clamp=True)
    with g_col2:
        st.write("**Fitted Scale Parameter (theta)**")
        st.caption(f"Mean: {res.theta_map.mean():.2f} | Min: {res.theta_map.min():.2f} | Max: {res.theta_map.max():.2f}")
        st.image(res.theta_map / (res.theta_map.max() + 1e-6), clamp=True)
    with g_col3:
        st.write("**Continuous Change Probability P_change**")
        st.caption("Non-parametric Upper-Tail Anomaly Likelihood")
        st.image(res.prob_change, clamp=True)

    st.write("---")
    st.markdown("### Graph Cut Unary Energy Data Terms")
    u1, u2 = st.columns(2)
    with u1:
        st.subheader("Unary Cost: Unchanged D_i(C_i = 0)")
        st.caption("-ln(Survival(Y_i)) - High cost penalized for change anomalies")
        st.image(res.d_unchanged / (res.d_unchanged.max() + 1e-6), clamp=True)
    with u2:
        st.subheader("Unary Cost: Changed D_i(C_i = 1)")
        st.caption("-ln(alpha) - Significance boundary baseline")
        st.image(res.d_changed / (res.d_changed.max() + 1e-6), clamp=True)


# TAB 5: Quantitative Evaluation & Diagnostics
with view_tabs[4]:
    st.markdown("### Module 5: Performance Evaluation & Accuracy Metrics")

    if eval_dict.get("has_ground_truth"):
        st.success("Ground Truth Reference Mask provided. Full quantitative confusion matrix evaluated.")

        metrics = eval_dict["metrics"]
        cm = eval_dict["confusion_matrix"]

        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                [
                    {"Class": "Predicted Changed (1)", "GT Changed (1)": cm["true_positive"], "GT Unchanged (0)": cm["false_positive"]},
                    {"Class": "Predicted Unchanged (0)", "GT Changed (1)": cm["false_negative"], "GT Unchanged (0)": cm["true_negative"]},
                ]
            ).set_index("Class")
            st.table(cm_df)

        with c_col2:
            st.subheader("Quantitative Accuracy Metrics")
            m_df = pd.DataFrame([
                {"Metric Name": "Overall Accuracy (OA)", "Score": f"{metrics['overall_accuracy']}%", "Description": "(TP + TN) / Total"},
                {"Metric Name": "F1-Score", "Score": f"{metrics['f1_score']}%", "Description": "Harmonic mean of Precision and Recall"},
                {"Metric Name": "Precision", "Score": f"{metrics['precision']}%", "Description": "TP / (TP + FP)"},
                {"Metric Name": "Recall (Detection Rate)", "Score": f"{metrics['recall']}%", "Description": "TP / (TP + FN)"},
                {"Metric Name": "Intersection over Union (IoU)", "Score": f"{metrics['iou']}%", "Description": "Jaccard Index"},
                {"Metric Name": "Commission Error (False Alarm)", "Score": f"{metrics['commission_error']}%", "Description": "FP / (TP + FP) = 1 - Precision"},
                {"Metric Name": "Omission Error (Missed)", "Score": f"{metrics['omission_error']}%", "Description": "FN / (TP + FN) = 1 - Recall"},
                {"Metric Name": "Cohen's Kappa", "Score": f"{metrics['cohens_kappa']}", "Description": "Chance-corrected inter-rater agreement"},
            ])
            st.table(m_df)

        # Ground Truth visualization
        if gt_input is not None:
            st.write("---")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.subheader("Reference Ground-Truth Change Mask")
                st.image((gt_input * 255).astype(np.uint8))
            with g_col2:
                st.subheader("Predicted vs Ground-Truth Error Map")
                # Green = TP, Red = FP, Blue = FN
                h, w = res.binary_change_map.shape
                error_vis = np.zeros((h, w, 3), dtype=np.uint8)
                pred_bin = res.binary_change_map
                gt_bin = cv2.resize(gt_input.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                error_vis[(pred_bin == 1) & (gt_bin == 1)] = [34, 197, 94]    # True Positive (Green)
                error_vis[(pred_bin == 1) & (gt_bin == 0)] = [239, 68, 68]    # False Positive / Commission (Red)
                error_vis[(pred_bin == 0) & (gt_bin == 1)] = [59, 130, 246]   # False Negative / Omission (Blue)

                st.image(error_vis)
                st.caption("Green: True Positive | Red: False Positive (Commission) | Blue: False Negative (Omission)")

    else:
        st.info("No ground truth mask was uploaded. Displaying observational summary statistics. (Validation metrics are strictly never fabricated without ground truth).")
        st.json(eval_dict)

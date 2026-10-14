# Land-Cover Change Detection using Adaptive Gamma-Graph Cut Spatial-Spectral Architecture (AGGC-SSDA) for Unsupervised Earth Observation Mapping

**Author:** Anuska Saha (Reg. No: 23BCE0167)  
**Affiliation:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, Tamil Nadu, India  
**Course:** Digital Image Processing (BCSE403L) — Review I  
**Faculty Guide:** Dr. Anisha M. Lal  

---

##  Project Overview & Motivation

Change detection in multitemporal satellite imagery is vital for urban expansion monitoring, forest canopy disturbance tracking, disaster damage assessment, and geospatial intelligence. Existing unsupervised change detection frameworks suffer from severe theoretical and operational limitations:

1. **Breakdown of Multivariate Normality & Orthogonality:** Chi-Square Transformation (CST; Ridd & Liu, 1998) assumes multi-band difference vectors strictly adhere to zero-mean multivariate normal distributions with mutual orthogonality. Real-world satellite scenes feature asymmetric, skewed, heavy-tailed noise, causing theoretical $\chi^2$ decision boundaries to produce high false-alarm commission errors.
2. **Spectral Index Saturation & Background Interference:** Vegetation indices like NDVI saturate in dense closed-canopy forests and are distorted by bare soil reflectance in arid/sparse canopy ecotones.
3. **Suboptimality of Rigid Global Thresholding:** Scene-wide histogram thresholds (Otsu, Kittler-Illingworth, $\mu \pm n\sigma$) fail under non-stationary spatial illumination, topography, and seasonal phenology shifts.
4. **Neglect of Spatial Context:** Per-pixel thresholding causes isolated salt-and-pepper noise artifacts, while traditional Object-Based Image Analysis (OBIA) multi-resolution segmentation blurs fine structural linear features (roads, rivers, deforestation boundaries).

### The Proposed AGGC-SSDA Architecture

**AGGC-SSDA** resolves these challenges by integrating:
- **Hybrid Radiometric Calibration** (Sub-pixel co-registration + Pseudo-Invariant Features RRN).
- **Soil-Resilient 4D Spatial-Spectral Feature Stack** combining visible red reflectance ($R_{vis}$), near-infrared ($NIR$), Tasseled Cap Wetness ($TC_{wetness}$), and GLCM Texture Entropy ($GLCM_{entropy}$).
- **Localized Two-Parameter Gamma Statistical Distribution Modeling** over localized covariance-weighted difference metrics $Y(x, y)$, replacing flawed Gaussian and $\chi^2$ assumptions.
- **Contrast-Weighted Graph-Cut MRF Energy Minimization** using the polynomial-time Boykov-Kolmogorov Max-Flow/Min-Cut algorithm.
- **Empirical Ground-Truth Validation** with zero fabricated numbers.

---

##  The Five Processing Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│ Module 1: Preprocessing & Radiometric Calibration                      │
│ - Channel & dimension verification                                    │
│ - Sub-pixel spatial co-registration (ECC / ORB homography)             │
│ - Relative Radiometric Normalization (PIF linear regression)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Module 2: Spatial-Spectral Feature Extraction                          │
│ - F(x, y) = [R_vis, NIR, TC_wetness, GLCM_entropy]^T                   │
│ - Multi-temporal difference: ΔF(x, y) = F_t2(x, y) - F_t1(x, y)        │
│ - Transparent RGB fallback mode with proxy indices                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Module 3: Localized Two-Parameter Gamma Statistical Modeling           │
│ - Localized covariance metric: Y = ΔF^T Σ_local^{-1} ΔF                │
│ - Local tile Gamma parameter estimation: f(Y; k, θ)                   │
│ - Unary data costs: D_i(0) = -ln(Survival(Y_i)), D_i(1) = -ln(α)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Module 4: Contrast-Weighted Graph-Cut Optimization                     │
│ - 8-connected grid graph G = (V, E)                                   │
│ - Pairwise edge capacity: w_ij = λ * exp(-||F(i)-F(j)||^2 / 2σ_s^2)    │
│ - Global energy minimization via Boykov-Kolmogorov Max-Flow/Min-Cut    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Module 5: Binary Change Map Generation & Evaluation                    │
│ - Optimal binary change map C(x, y) ∈ {0, 1}                          │
│ - Translucent color overlay with sharp boundary contours               │
│ - Morphological patch analytics & Ground-Truth validation metrics     │
└────────────────────────────────────────────────────────────────────────┘
```

---

##  Mathematical Formulation

### 1. Multi-Feature Stack Difference
$$F(x, y) = [R_{vis}(x, y), NIR(x, y), TC_{wetness}(x, y), GLCM_{entropy}(x, y)]^T$$
$$\Delta F(x, y) = F_{t2}(x, y) - F_{t1}(x, y)$$

### 2. Localized Distance Metric
$$Y(x, y) = \Delta F(x, y)^T \Sigma_{local}^{-1}(x, y) \Delta F(x, y)$$
where $\Sigma_{local}$ is the localized background noise covariance estimated from inliers in an $N \times N$ spatial neighborhood.

### 3. Two-Parameter Gamma Probability Density
$$f(Y; k, \theta) = \frac{1}{\theta^k \Gamma(k)} Y^{k-1} e^{-Y/\theta}, \quad Y > 0, k > 0, \theta > 0$$
where $k$ is the shape parameter and $\theta$ is the scale parameter estimated locally across spatial sub-regions.

### 4. Graph-Cut Energy Function
$$E(C) = \sum_{i \in V} D_i(C_i) + \lambda \sum_{(i,j) \in E} V_{ij}(C_i, C_j)$$
- **Unary Term:**
  $$D_i(C_i = 0) = -\ln(1 - \int_0^{Y_i} f(t; k_i, \theta_i) dt) = -\ln(\text{Survival}(Y_i))$$
  $$D_i(C_i = 1) = -\ln(\alpha)$$
- **Pairwise Term:**
  $$V_{ij}(C_i, C_j) = |C_i - C_j| \cdot \exp\left(-\frac{||F_{t1}(i) - F_{t1}(j)||^2}{2\sigma_s^2}\right)$$

---

##  Important Note on Submission PDF Diagram Discrepancy

On Page 16 of the submitted Review I PDF, an inserted graphic is titled *"Hybrid Digital Image Processing Framework for Smart Road Damage Detection and Severity Analysis"* (referencing potholes, crack detection, and CLAHE). As noted in project requirements, this graphic is an accidental insertion from a civil engineering domain. 

This software implementation strictly and authentically adheres to the **AGGC-SSDA Earth Observation Change Detection Architecture** detailed across all textual descriptions, equations, and literature reviews in the submission.

---

##  Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 1. Clone or Navigate to Project Directory
```powershell
cd "c:\Users\Anuska saha\OneDrive\Digital image processing project"
```

### 2. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

##  Automated Test Suite

Run the full automated test suite covering all 5 modules and end-to-end integration:

```powershell
python -m pytest -v
```

### Verified Test Coverage
- `tests/test_preprocessing.py`: Dimension checking, normalization, sub-pixel registration, PIF RRN.
- `tests/test_features.py`: GLCM entropy, 4D feature extraction (multispectral & RGB fallback), difference computation.
- `tests/test_gamma_modeling.py`: Covariance distance $Y$, localized Gamma distribution fitting, unary log-survival costs.
- `tests/test_graph_cut.py`: 8-connected grid contrast weights, PyMaxflow min-cut optimization.
- `tests/test_evaluation.py`: Confusion matrix, OA, F1, precision, recall, commission/omission errors, overlay blending.
- `tests/test_pipeline_integration.py`: End-to-end 5-module execution on synthetic multitemporal benchmark scenes.

---

##  Running the Interactive Streamlit Web Application

Launch the Streamlit web dashboard:

```powershell
python -m streamlit run app.py
```

### Dashboard Features
- **Input Flexibility**: Choose between the built-in documented benchmark satellite pair (deforestation & urban expansion) or upload custom multitemporal images ($T_1$, $T_2$, and optional ground truth).
- **Interactive Parameter Tuning**: Adjust localized covariance window $N$, spatial tile size, significance cutoff $\alpha$, spatial smoothness $\lambda$, and contrast scale $\sigma_s$.
- **Real-Time Visualization Tabs**:
  - **Final Results**: High-contrast binary change map $C(x, y)$ and customizable translucent color overlay.
  - **Multitemporal Acquisitions**: Side-by-side comparison with registration & PIF normalization telemetry.
  - **Intermediate Features**: Inspection of Red reflectance, NIR/proxy, Tasseled Cap Wetness, GLCM entropy, and distance scalar $Y(x, y)$.
  - **Gamma Statistics**: Localized shape $k$ and scale $\theta$ parameter maps, continuous change probabilities, and unary cost distributions.
  - **Quantitative Evaluation**: Confusion matrix ($TP, FP, TN, FN$), Overall Accuracy, F1-Score, Precision, Recall, IoU, Commission Error, Omission Error, and error map visualization (when ground truth is supplied).
- **Artifact Downloads**: Export binary change masks (PNG), overlays (PNG), summary tables (CSV), and complete diagnostics (JSON).

---

##  Project Structure

```
.
├── aggc_ssda/                     # Core algorithmic package
│   ├── __init__.py                # Package exports
│   ├── preprocessing.py           # Module 1: Co-registration & RRN (PIF)
│   ├── features.py                # Module 2: 4D Spectral-Spatial & GLCM features
│   ├── gamma_modeling.py          # Module 3: Localized Gamma estimation & unary terms
│   ├── graph_cut.py               # Module 4: Contrast-weighted MRF Graph-Cut solver
│   ├── evaluation.py              # Module 5: Metrics, morphology & overlay generator
│   ├── pipeline.py                # Pipeline orchestrator
│   └── dataset_loader.py          # Procedural benchmark generator & IO helpers
│
├── data/
│   └── sample/                    # Documented benchmark Earth Observation dataset
│       ├── t1_pre.png             # Pre-event acquisition
│       ├── t2_post.png            # Post-event acquisition
│       ├── ground_truth.png       # Reference binary change mask
│       └── README.md              # Dataset documentation & limitations
│
├── tests/                         # Automated unit & integration test suite
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_gamma_modeling.py
│   ├── test_graph_cut.py
│   ├── test_evaluation.py
│   └── test_pipeline_integration.py
│
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Pinned package dependencies
└── README.md                      # Comprehensive documentation
```

---

##  Academic Integrity Statement
All quantitative metrics reported by this system (Precision, Recall, F1, OA, Commission/Omission errors) are computed **strictly from user-supplied ground truth masks**. When no ground truth is provided, the software transparently displays observational statistics and indicates that validation metrics were not evaluated.

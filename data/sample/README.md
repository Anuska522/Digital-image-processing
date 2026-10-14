# Sample Multitemporal Satellite Dataset

This folder contains a benchmark multitemporal Earth Observation dataset pair with an authentic binary ground-truth change mask for testing and demonstration purposes.

## File Inventory
- **`t1_pre.png`**: Pre-event reference acquisition (256×256, 3-channel RGB). Depicts natural land-cover comprising dense canopy forest, meandering water body, bare semi-arid soil, and agricultural plots.
- **`t2_post.png`**: Post-event multitemporal acquisition (256×256, 3-channel RGB). Captures real land-cover conversions plus atmospheric illumination gradients and sensor noise.
- **`ground_truth.png`**: Exact binary reference change mask (256×256, 1-channel, 0 = Unchanged, 255 = Changed).

## Ground Truth Provenance & Simulated Conversions
- **Deforestation / Canopy Loss**: Clear-cut forest boundary in the upper-left quadrant converted from dark green chlorophyll canopy to bare, reflective soil.
- **Urban Development**: New high-reflectance structural footprint in the upper-right quadrant representing construction expansion.
- **Invariant Background Variations**: Includes a non-uniform spatial illumination gradient across the scene and subtle natural agricultural variations to test the resilience of localized Gamma statistical modeling and contrast-weighted graph-cut smoothing against false alarms.

## Limitations
- **Resolution**: 256×256 pixels for fast, lightweight testing and demonstration.
- **Spectral Bands**: 3-band visible RGB format (triggers Module 2's transparent RGB fallback mode). For native multispectral testing (Sentinel-2, Landsat-8), users can upload 4+ band GeoTIFF images directly through the Streamlit web dashboard.

"""
Command-Line Interface (CLI) Entrypoint for AGGC-SSDA.

Allows execution of the 5-module change detection pipeline from terminal / shell:
    python -m aggc_ssda --benchmark
    python -m aggc_ssda --t1 path/to/t1.png --t2 path/to/t2.png --output ./results
"""

import argparse
import sys
import os
import json
import numpy as np

from .pipeline import AGGCSSDAPipeline
from .dataset_loader import load_image_from_file, save_image_to_file, generate_benchmark_satellite_pair


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="AGGC-SSDA: Adaptive Gamma-Graph Cut Spatial-Spectral Architecture for Change Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--t1", type=str, default=None, help="Path to pre-event reference image (T1)")
    parser.add_argument("--t2", type=str, default=None, help="Path to post-event subject image (T2)")
    parser.add_argument("--gt", type=str, default=None, help="Path to optional ground-truth binary mask")
    parser.add_argument("--output", "-o", type=str, default=None, help="Directory to save generated artifacts")
    parser.add_argument("--benchmark", action="store_true", help="Run on procedurally synthesized benchmark pair")
    parser.add_argument("--alpha", type=float, default=0.005, help="Statistical significance threshold alpha")
    parser.add_argument("--lambda-param", type=float, default=2.0, help="Spatial smoothness regularization lambda")
    parser.add_argument("--tile-size", type=int, default=64, help="Spatial tile size for localized Gamma estimation")
    parser.add_argument("--no-coreg", action="store_true", help="Disable sub-pixel co-registration")
    parser.add_argument("--no-rrn", action="store_true", help="Disable Relative Radiometric Normalization")
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)

    if parsed.benchmark or (parsed.t1 is None and parsed.t2 is None):
        print("[INFO] Generating synthetic Earth Observation benchmark dataset...")
        t1, t2, gt, _ = generate_benchmark_satellite_pair(256, 256)
    else:
        if parsed.t1 is None or parsed.t2 is None:
            print("[ERROR] Both --t1 and --t2 must be provided unless using --benchmark.", file=sys.stderr)
            sys.exit(1)
        t1 = load_image_from_file(parsed.t1)
        t2 = load_image_from_file(parsed.t2)
        gt = load_image_from_file(parsed.gt) if parsed.gt else None
        if gt is not None:
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt = (gt > (127 if gt.max() > 1 else 0.5)).astype(np.uint8)

    print(f"[INFO] Initializing AGGC-SSDA Pipeline (alpha={parsed.alpha}, lambda={parsed.lambda_param}, tile_size={parsed.tile_size})...")
    pipeline = AGGCSSDAPipeline(
        do_coregistration=not parsed.no_coreg,
        do_radiometric_norm=not parsed.no_rrn,
        alpha_significance=parsed.alpha,
        lambda_param=parsed.lambda_param,
        tile_size=parsed.tile_size,
    )

    print("[INFO] Executing 5-module change detection workflow...")
    result = pipeline.run(t1_img=t1, t2_img=t2, ground_truth_mask=gt)

    metrics = result.evaluation_metrics
    print("\n" + "=" * 50)
    print("           AGGC-SSDA EXECUTION REPORT            ")
    print("=" * 50)
    print(f"Total Pixels        : {metrics.get('total_pixels', 0):,}")
    print(f"Changed Pixels      : {metrics.get('changed_pixels', 0):,}")
    print(f"Change Percentage   : {metrics.get('change_percentage', 0.0)}%")
    print(f"Change Patches      : {metrics.get('morphology', {}).get('patch_count', 0)}")
    print(f"Execution Time      : {result.execution_time_seconds:.3f} s")

    if metrics.get("has_ground_truth"):
        m = metrics["metrics"]
        print("-" * 50)
        print("          GROUND-TRUTH VALIDATION METRICS         ")
        print("-" * 50)
        print(f"Overall Accuracy    : {m['overall_accuracy']}%")
        print(f"F1-Score            : {m['f1_score']}%")
        print(f"Precision           : {m['precision']}%")
        print(f"Recall              : {m['recall']}%")
        print(f"IoU (Jaccard)       : {m['iou']}%")
        print(f"Commission Error    : {m['commission_error']}%")
        print(f"Omission Error      : {m['omission_error']}%")
        print(f"Cohen's Kappa       : {m['cohens_kappa']}")

    print("=" * 50)

    if parsed.output:
        out_dir = os.path.abspath(parsed.output)
        os.makedirs(out_dir, exist_ok=True)
        mask_path = os.path.join(out_dir, "change_map.png")
        overlay_path = os.path.join(out_dir, "change_overlay.png")
        meta_path = os.path.join(out_dir, "diagnostics.json")

        save_image_to_file((result.binary_change_map * 255).astype(np.uint8), mask_path)
        save_image_to_file(result.change_overlay, overlay_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result.module_diagnostics, f, indent=2)

        print(f"\n[INFO] Artifacts successfully exported to: {out_dir}")

    return result


if __name__ == "__main__":
    main()

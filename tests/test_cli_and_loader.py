"""
Unit and integration tests for dataset loader and CLI entrypoint.
"""

import os
import tempfile
import pytest
import numpy as np

from aggc_ssda.dataset_loader import (
    load_image_from_file,
    save_image_to_file,
    generate_benchmark_satellite_pair,
)
from aggc_ssda.__main__ import main, parse_args


def test_dataset_loader_roundtrip_tiff_multiband():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 6-band raster (e.g. Landsat-8)
        img_6b = (np.random.uniform(0.1, 0.9, (32, 32, 6)) * 255).astype(np.uint8)
        path = os.path.join(tmp_dir, "test_multispectral.tif")

        save_image_to_file(img_6b, path)
        loaded = load_image_from_file(path)

        assert loaded.shape == (32, 32, 6)
        assert np.array_equal(img_6b, loaded)


def test_dataset_loader_roundtrip_png():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_rgb = np.full((24, 24, 3), 128, dtype=np.uint8)
        path = os.path.join(tmp_dir, "test_image.png")

        save_image_to_file(img_rgb, path)
        loaded = load_image_from_file(path)

        assert loaded.shape == (24, 24, 3)
        assert np.allclose(loaded, 128, atol=2)


def test_dataset_loader_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_image_from_file("non_existent_file_path_12345.png")


def test_save_image_to_file_empty():
    with pytest.raises(ValueError, match="empty"):
        save_image_to_file(np.empty((0, 0), dtype=np.uint8), "out.png")


def test_cli_benchmark_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "benchmark_results")
        res = main(["--benchmark", "--output", out_dir, "--alpha", "0.01"])

        assert res.binary_change_map.shape == (256, 256)
        assert os.path.exists(os.path.join(out_dir, "change_map.png"))
        assert os.path.exists(os.path.join(out_dir, "change_overlay.png"))
        assert os.path.exists(os.path.join(out_dir, "diagnostics.json"))


def test_cli_missing_pair_arguments():
    with pytest.raises(SystemExit):
        main(["--t1", "only_one_file.png"])

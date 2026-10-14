from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import pyarrow.parquet as pq

from pelagic_benchmark.analysis.matching import match_components
from pelagic_benchmark.analysis.pixels import threshold_crossings
from pelagic_benchmark.analysis.preprocessing import apply_preprocessing
from pelagic_benchmark.analysis.segmentation import segment
from pelagic_benchmark.codecs import codec_for
from pelagic_benchmark.dataset import pixel_hash
from pelagic_benchmark.runner import run
from pelagic_benchmark.config import load_config
from pelagic_benchmark.config import PreprocessingConfig
from pelagic_benchmark.verify import verify
from pelagic_benchmark.system_benchmarks import run_system_benchmarks
from pelagic_benchmark.config import SystemBenchmarkConfig


def test_canonical_hash_is_stable_and_shape_aware() -> None:
    pixels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert pixel_hash(pixels) == pixel_hash(pixels.copy())
    assert pixel_hash(pixels) != pixel_hash(pixels.reshape(1, 4))


def test_inversion_happens_after_preprocessing() -> None:
    pixels = np.array([[0, 1], [127, 255]], dtype=np.uint8)
    condition = PreprocessingConfig("flatfield_window_column_mean", True, flatfield_axis=0)
    corrected, _ = apply_preprocessing(pixels, condition)
    inverted, metadata = apply_preprocessing(pixels, condition, invert_grayscale=True)
    assert np.array_equal(inverted, np.bitwise_not(corrected))
    raw_inverted, _ = apply_preprocessing(pixels, PreprocessingConfig("raw", False), invert_grayscale=True)
    assert np.array_equal(raw_inverted, np.array([[255, 254], [128, 0]], dtype=np.uint8))
    assert inverted.dtype == np.uint8 and metadata["inversion_stage"] == "after_preprocessing"


def test_png_is_exactly_lossless() -> None:
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    codec = codec_for("png")
    assert np.array_equal(codec.decode(codec.encode(pixels, {})), pixels)


def test_jpeg_shape_dtype_and_stable_treatment_ids() -> None:
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    codec = codec_for("jpeg")
    treatments = codec.treatments([95, 75])
    assert [item.setting_id for item in treatments] == ["jpeg_q95", "jpeg_q75"]
    decoded = codec.decode(codec.encode(pixels, treatments[0].settings))
    assert decoded.shape == pixels.shape and decoded.dtype == np.uint8


@pytest.mark.parametrize(
    ("name", "options", "exact_setting"),
    [
        ("png", {}, "png_deflate_l6"),
        ("tiff", {"compressions": ["none", "deflate", "zstd"]}, "tiff_zstd_l6"),
        ("jpeg2000", {"lossless": True}, "jpeg2000_lossless"),
        ("jpegxl", {"lossless": True}, "jpegxl_lossless"),
        ("webp", {"lossless": True}, "webp_lossless"),
        ("jpegls", {"lossless": True}, "jpegls_lossless"),
    ],
)
def test_requested_lossless_codecs_round_trip_exactly(name, options, exact_setting) -> None:
    pixels = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    codec = codec_for(name)
    treatment = next(item for item in codec.treatments(None, options) if item.setting_id == exact_setting)
    assert np.array_equal(codec.decode(codec.encode(pixels, treatment.settings)), pixels)


@pytest.mark.parametrize(
    ("name", "values", "options", "expected"),
    [
        ("jpeg2000", None, {"lossless": False, "lossy_levels": [50]}, "jpeg2000_l50"),
        ("jpegxl", None, {"lossless": False, "distances": [1.0]}, "jpegxl_d1p0"),
        ("webp", [75], {"lossless": False}, "webp_q75"),
        ("avif", None, {"quality_levels": [40]}, "avif_q40"),
        ("jpegls", None, {"lossless": False, "near_lossless": [3]}, "jpegls_near3"),
    ],
)
def test_requested_lossy_codecs_preserve_grayscale_shape(name, values, options, expected) -> None:
    pixels = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    codec = codec_for(name)
    treatment = codec.treatments(values, options)[0]
    assert treatment.setting_id == expected
    decoded = codec.decode(codec.encode(pixels, treatment.settings))
    assert decoded.shape == pixels.shape and decoded.dtype == np.uint8


def test_fixed_segmentation_and_threshold_crossings() -> None:
    pixels = np.array([[0, 119, 120], [121, 10, 122]], dtype=np.uint8)
    result = segment(pixels, "fixed", 118, 1)
    assert result.mask.tolist() == [[False, True, True], [True, False, True]]
    changed, fraction = threshold_crossings(np.array([[117, 118, 119]], dtype=np.uint8), np.array([[119, 117, 119]], dtype=np.uint8), 118)
    assert (changed, fraction) == (2, pytest.approx(2 / 3))


def test_pelagia_profile_flatfield_condition_is_named_and_uint8() -> None:
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    corrected, metadata = apply_preprocessing(
        pixels,
        PreprocessingConfig("flatfield_window_column_mean", True, flatfield_axis=0),
    )
    assert corrected.dtype == np.uint8 and corrected.shape == pixels.shape
    assert not np.array_equal(corrected, pixels)
    assert metadata["preprocessing_id"] == "flatfield_window_column_mean"
    assert metadata["engine"] == "Pelagia.processing.frame_correction.flatfield_profile_correction"


def test_system_context_benchmarks_are_temporary(tmp_path: Path) -> None:
    rows = run_system_benchmarks(SystemBenchmarkConfig(True, tmp_path / "system", [1], 1, 16), "test-run")
    assert {(row["category"], row["operation"]) for row in rows} == {("disk", "write"), ("disk", "fsync"), ("disk", "read"), ("numpy", "matmul_float32"), ("numpy", "rfft2_float32")}
    assert not (tmp_path / "system" / "test-run").exists()


def _seg(mask: list[list[int]]):
    return segment(np.array(mask, dtype=np.uint8) * 255, "fixed", 118, 1)


@pytest.mark.parametrize(
    ("reference", "treatment", "expected"),
    [
        ([[1, 1, 0], [1, 1, 0]], [[1, 1, 0], [1, 1, 0]], "1:1"),
        ([[1, 1, 1]], [[1, 0, 1]], "split"),
        ([[1, 0, 1]], [[1, 1, 1]], "merge"),
        ([[1, 1]], [[0, 0]], "lost"),
        ([[0, 0]], [[1, 1]], "new"),
    ],
)
def test_component_relationships(reference, treatment, expected) -> None:
    assert expected in {row["relationship"] for row in match_components(_seg(reference), _seg(treatment))}


def test_iou_is_correct() -> None:
    rows = match_components(_seg([[1, 1, 0]]), _seg([[0, 1, 1]]))
    assert rows[0]["relationship"] == "1:1"
    assert rows[0]["intersection_pixels"] == 1
    assert rows[0]["union_pixels"] == 3
    assert rows[0]["iou"] == pytest.approx(1 / 3)


def test_complete_experiment_and_verify(tmp_path: Path) -> None:
    dataset = tmp_path / "images"
    dataset.mkdir()
    Image.fromarray(np.array([[0, 0, 0, 0], [0, 150, 150, 0], [0, 150, 150, 0], [0, 0, 0, 0]], dtype=np.uint8)).save(dataset / "one.png")
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(f'''schema_version = 1
[dataset]
path = "{dataset}"
invert_grayscale = true
[segmentation]
methods = ["fixed", "otsu"]
fixed_threshold = 118
minimum_area = 1
[[codecs]]
name = "png"
[[codecs]]
name = "jpeg"
qualities = [95, 75]
[benchmark]
warmup_runs = 0
timed_runs = 1
in_memory_enabled = true
roundtrip_enabled = true
roundtrip_directory = "{tmp_path / 'scratch'}"
[output]
directory = "{tmp_path / 'runs'}"
''')
    events: list[str] = []
    def progress(event, payload):
        events.append(event)
        # Simulate an external cleanup race during a long run. The immutable
        # snapshot must be restored before manifest finalization.
        if event == "frame":
            for snapshot in (tmp_path / "runs").glob("*/config.toml"):
                snapshot.unlink()
    output = run(load_config(config_path), progress=progress)
    assert {path.name for path in output.glob("*.parquet")} == {f"{name}.parquet" for name in ("images", "preprocessing_results", "system_benchmark_results", "codec_results", "benchmark_results", "pixel_results", "segmentation_results", "roi_matches", "errors")}
    assert len(verify(output)) == 9
    preprocessing = pq.read_table(output / "preprocessing_results.parquet").to_pylist()
    assert preprocessing[0]["preprocessing_id"] == "raw"
    source_pixels = np.array([[0, 0, 0, 0], [0, 150, 150, 0], [0, 150, 150, 0], [0, 0, 0, 0]], dtype=np.uint8)
    expected_reference = np.bitwise_not(source_pixels)
    images = pq.read_table(output / "images.parquet").to_pylist()
    assert images[0]["pixel_sha256"] == pixel_hash(source_pixels)
    assert preprocessing[0]["reference_pixel_sha256"] == pixel_hash(expected_reference)
    assert json.loads((output / "manifest.json").read_text())["canonical_representation"]["transform"] == "inverse_uint8_after_preprocessing"
    benchmark_rows = pq.read_table(output / "benchmark_results.parquet").to_pylist()
    assert {(row["benchmark_kind"], row["operation"]) for row in benchmark_rows} >= {("in_memory", "encode"), ("in_memory", "decode"), ("roundtrip", "write"), ("roundtrip", "read"), ("roundtrip", "decode")}
    assert not list((tmp_path / "scratch").rglob("*.bin"))
    assert events[0] == "started" and events[-1] == "completed"
    assert events.count("frame") == 1 and events.count("treatment") == 3
    (output / "images.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify(output)
    (output / "images.parquet").unlink()
    with pytest.raises(ValueError, match="Missing artifact"):
        verify(output)

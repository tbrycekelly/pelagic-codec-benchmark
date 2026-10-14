from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMAS: dict[str, pa.Schema] = {
    "images": pa.schema([( "image_id", pa.string()), ("source_path", pa.string()), ("source_frame_index", pa.int64()), ("source_sha256", pa.string()), ("pixel_sha256", pa.string()), ("width", pa.int64()), ("height", pa.int64()), ("dtype", pa.string()), ("pixel_count", pa.int64())]),
    "preprocessing_results": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("settings_json", pa.string()), ("reference_pixel_sha256", pa.string()), ("engine", pa.string())]),
    "system_benchmark_results": pa.schema([( "run_id", pa.string()), ("category", pa.string()), ("operation", pa.string()), ("setting", pa.string()), ("iteration", pa.int64()), ("byte_count", pa.int64()), ("elapsed_ns", pa.int64()), ("throughput_mib_per_second", pa.float64()), ("cache_state", pa.string())]),
    "codec_results": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("settings_json", pa.string()), ("input_pixel_bytes", pa.int64()), ("encoded_bytes", pa.int64()), ("bits_per_pixel", pa.float64()), ("compression_ratio", pa.float64()), ("lossless", pa.bool_())]),
    "benchmark_results": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("benchmark_kind", pa.string()), ("operation", pa.string()), ("iteration", pa.int64()), ("elapsed_ns", pa.int64()), ("megapixels_per_second", pa.float64())]),
    "pixel_results": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("mae", pa.float64()), ("rmse", pa.float64()), ("max_absolute_error", pa.int64()), ("psnr", pa.float64()), ("exact_equal", pa.bool_())]),
    "segmentation_results": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("segmentation_method", pa.string()), ("threshold", pa.float64()), ("reference_threshold", pa.float64()), ("treatment_threshold", pa.float64()), ("threshold_delta", pa.float64()), ("object_count", pa.int64()), ("foreground_pixel_count", pa.int64()), ("foreground_fraction", pa.float64()), ("threshold_crossing_pixels", pa.int64()), ("threshold_crossing_fraction", pa.float64()), ("engine", pa.string())]),
    "roi_matches": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("preprocessing_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("segmentation_method", pa.string()), ("relationship", pa.string()), ("reference_component_id", pa.int64()), ("treatment_component_id", pa.int64()), ("intersection_pixels", pa.int64()), ("union_pixels", pa.int64()), ("iou", pa.float64()), ("reference_area", pa.int64()), ("treatment_area", pa.int64()), ("relative_area_error", pa.float64()), ("centroid_dx", pa.float64()), ("centroid_dy", pa.float64()), ("centroid_displacement", pa.float64()), ("reference_perimeter", pa.float64()), ("treatment_perimeter", pa.float64()), ("reference_bbox", pa.string()), ("treatment_bbox", pa.string())]),
    "errors": pa.schema([( "run_id", pa.string()), ("image_id", pa.string()), ("codec", pa.string()), ("setting_id", pa.string()), ("stage", pa.string()), ("exception_type", pa.string()), ("message", pa.string())]),
}


def write_table(directory: Path, name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    path = directory / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), path, compression="zstd")
    return {"sha256": sha256(path), "rows": len(rows), "schema": str(SCHEMAS[name])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

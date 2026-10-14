from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable

from .analysis.matching import match_components
from .analysis.pixels import pixel_metrics, threshold_crossings
from .analysis.preprocessing import apply_preprocessing
from .analysis.segmentation import segment
from .codecs import codec_for
from .config import ExperimentConfig
from .dataset import discover_images, discover_videos, estimate_video_frame_count, iter_video_frames, load_canonical
from .environment import provenance
from .storage import SCHEMAS, sha256, write_json, write_table
from .system_benchmarks import run_system_benchmarks


ProgressCallback = Callable[[str, dict[str, object]], None]


def run(config: ExperimentConfig, progress: ProgressCallback | None = None) -> Path:
    started = datetime.now(UTC)
    run_id = f"{started.strftime('%Y-%m-%dT%H%M%S%fZ')}_{config.sha256[:7]}"
    output = config.output_directory / run_id
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config.toml"
    config_snapshot.write_bytes(config.raw)
    rows = {name: [] for name in SCHEMAS}
    codec_metadata: dict[str, dict[str, str]] = {}
    if config.dataset_type == "images":
        source_paths = discover_images(config.dataset_path)
        estimated_frames = len(source_paths)
        input_groups = ((path, iter((load_canonical(path, config.dataset_path),))) for path in source_paths)
    else:
        source_paths = discover_videos(config.dataset_path)
        estimated_frames = estimate_video_frame_count(source_paths) if config.frame_selection == "all" else len(source_paths) * len(config.frame_indices)
        input_groups = ((path, iter_video_frames(path, config.dataset_path, config.frame_selection, config.frame_indices)) for path in source_paths)
    treatments_per_image = len(config.preprocessing) * sum(len(codec_for(item.name).treatments(item.qualities, item.options)) for item in config.codecs)
    estimated_system_observations = (len(config.system_benchmarks.disk_file_sizes_mib or []) * config.system_benchmarks.timed_runs * 3 + config.system_benchmarks.timed_runs * 2) if config.system_benchmarks.enabled else 0
    if progress:
        progress("started", {"estimated_frames": estimated_frames, "estimated_treatments": estimated_frames * treatments_per_image, "estimated_system_observations": estimated_system_observations, "source_count": len(source_paths)})
    scratch_directory = config.roundtrip_directory / run_id if config.roundtrip_enabled and config.roundtrip_directory else None
    if scratch_directory:
        scratch_directory.mkdir(parents=True, exist_ok=False)
    for source_path, images in input_groups:
        try:
            for image in images:
                _process_image(image, config, run_id, output, rows, codec_metadata, scratch_directory, progress)
                if progress:
                    progress("frame", {"image_id": image.image_id})
        except Exception as exc:
            rows["errors"].append(_error(run_id, None, None, None, "canonical_decode", exc))
    if scratch_directory is not None:
        try:
            scratch_directory.rmdir()
        except OSError as exc:
            rows["errors"].append(_error(run_id, None, None, None, "roundtrip_cleanup", exc))
    try:
        rows["system_benchmark_results"].extend(run_system_benchmarks(config.system_benchmarks, run_id))
        if progress:
            progress("system", {"observations": len(rows["system_benchmark_results"])})
    except Exception as exc:
        rows["errors"].append(_error(run_id, None, None, None, "system_benchmarks", exc))
    artifacts = {name: write_table(output, name, table_rows) for name, table_rows in rows.items()}
    # The run owns this immutable copy. Recreate it if a transient external
    # cleanup process removed it during a long execution.
    if not config_snapshot.is_file() or config_snapshot.read_bytes() != config.raw:
        config_snapshot.write_bytes(config.raw)
    manifest = {"schema_version": 1, "run_id": run_id, "status": "completed", "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(), "config_sha256": config.sha256, "config_file_sha256": config.sha256, "provenance": provenance(Path.cwd()), "canonical_representation": {"source_dtype": "uint8", "source_dimensions": 2, "transform": "inverse_uint8_after_preprocessing" if config.invert_grayscale else "identity"}, "codec_implementations": codec_metadata, "artifacts": artifacts, "segmentation": {"foreground_polarity": "bright", "segmentation_comparison": "pixel > threshold", "threshold_crossing_comparison": "pixel < threshold", "pelagia_adapter": "Pelagia threshold functions when installed; OpenCV fallback otherwise"}, "storage": {"benchmark_codec_path": "in-memory and optional filesystem roundtrip", "pelagia_storage_used": False, "retained_artifacts": config.retain_artifacts}, "system_benchmarks": {"enabled": config.system_benchmarks.enabled, "cache_semantics": "disk reads are warm-after-write; fsync is recorded separately"}}
    write_json(output / "manifest.json", manifest)
    if progress:
        progress("completed", {"frames": len(rows["images"]), "treatments": len(rows["codec_results"]), "errors": len(rows["errors"])})
    return output


def _process_image(image, config, run_id, output, rows, codec_metadata, scratch_directory, progress) -> None:
    pixels = image.pixels
    rows["images"].append({"image_id": image.image_id, "source_path": image.source_path, "source_frame_index": image.source_frame_index, "source_sha256": image.source_sha256, "pixel_sha256": image.pixel_sha256, "width": pixels.shape[1], "height": pixels.shape[0], "dtype": str(pixels.dtype), "pixel_count": pixels.size})
    for condition in config.preprocessing:
        reference_pixels, preprocessing = apply_preprocessing(pixels, condition, config.invert_grayscale)
        preprocessing_id = condition.id
        rows["preprocessing_results"].append({"run_id": run_id, "image_id": image.image_id, "preprocessing_id": preprocessing_id, "settings_json": json.dumps(preprocessing, sort_keys=True, separators=(",", ":")), "reference_pixel_sha256": preprocessing["pixel_sha256"], "engine": preprocessing["engine"]})
        reference_segments = ({method: segment(reference_pixels, method, config.fixed_threshold, config.minimum_area) for method in config.methods} if condition.segmentation_enabled else {})
        for method, reference in reference_segments.items():
            _add_segmentation(rows["segmentation_results"], run_id, image.image_id, preprocessing_id, "reference", "reference", method, reference, reference, reference_pixels, reference_pixels)
        for codec_config in config.codecs:
            codec = codec_for(codec_config.name)
            codec_metadata[codec.name] = codec.implementation_metadata()
            for treatment in codec.treatments(codec_config.qualities, codec_config.options):
                try:
                    data, timing_rows = _time_codec(codec, reference_pixels, treatment.settings, config.warmup_runs, config.timed_runs, run_id, image.image_id, preprocessing_id, treatment.codec, treatment.setting_id, config.in_memory_enabled, scratch_directory)
                    rows["benchmark_results"].extend(timing_rows)
                    decoded = codec.decode(data)
                    input_bytes = reference_pixels.nbytes
                    rows["codec_results"].append({"run_id": run_id, "image_id": image.image_id, "preprocessing_id": preprocessing_id, "codec": treatment.codec, "setting_id": treatment.setting_id, "settings_json": json.dumps(treatment.settings, sort_keys=True, separators=(",", ":")), "input_pixel_bytes": input_bytes, "encoded_bytes": len(data), "bits_per_pixel": 8 * len(data) / reference_pixels.size, "compression_ratio": input_bytes / len(data), "lossless": treatment.lossless})
                    if condition.pixel_metrics_enabled:
                        metrics = pixel_metrics(reference_pixels, decoded)
                        rows["pixel_results"].append({"run_id": run_id, "image_id": image.image_id, "preprocessing_id": preprocessing_id, "codec": treatment.codec, "setting_id": treatment.setting_id, **metrics})
                    for method, reference in reference_segments.items():
                        treatment_segment = segment(decoded, method, config.fixed_threshold, config.minimum_area)
                        _add_segmentation(rows["segmentation_results"], run_id, image.image_id, preprocessing_id, treatment.codec, treatment.setting_id, method, reference, treatment_segment, reference_pixels, decoded)
                        for match in match_components(reference, treatment_segment):
                            rows["roi_matches"].append({"run_id": run_id, "image_id": image.image_id, "preprocessing_id": preprocessing_id, "codec": treatment.codec, "setting_id": treatment.setting_id, "segmentation_method": method, **match})
                    if config.retain_artifacts:
                        _write_artifacts(output, image.image_id, f"{preprocessing_id}_{treatment.setting_id}", data, decoded)
                except Exception as exc:
                    rows["errors"].append(_error(run_id, image.image_id, treatment.codec, treatment.setting_id, "treatment", exc))
                finally:
                    if progress:
                        progress("treatment", {"image_id": image.image_id, "preprocessing_id": preprocessing_id, "codec": treatment.codec, "setting_id": treatment.setting_id})


def _time_codec(codec, pixels, settings, warmups, timed, run_id, image_id, preprocessing_id, codec_name, setting_id, in_memory_enabled, scratch_directory):
    if in_memory_enabled:
        for _ in range(warmups):
            warm = codec.encode(pixels, settings)
            codec.decode(warm)
    encoded: bytes | None = None
    rows: list[dict[str, object]] = []
    megapixels = pixels.size / 1_000_000
    for iteration in range(timed):
        if in_memory_enabled:
            started = time.perf_counter_ns()
            encoded = codec.encode(pixels, settings)
            elapsed = time.perf_counter_ns() - started
            rows.append(_timing(run_id, image_id, preprocessing_id, codec_name, setting_id, "in_memory", "encode", iteration, elapsed, megapixels))
            started = time.perf_counter_ns()
            codec.decode(encoded)
            elapsed = time.perf_counter_ns() - started
            rows.append(_timing(run_id, image_id, preprocessing_id, codec_name, setting_id, "in_memory", "decode", iteration, elapsed, megapixels))
        else:
            encoded = codec.encode(pixels, settings)
        if scratch_directory is not None:
            artifact = scratch_directory / f"{image_id}_{preprocessing_id}_{setting_id}_{iteration}.bin"
            try:
                started = time.perf_counter_ns()
                artifact.write_bytes(encoded)
                elapsed = time.perf_counter_ns() - started
                rows.append(_timing(run_id, image_id, preprocessing_id, codec_name, setting_id, "roundtrip", "write", iteration, elapsed, megapixels))
                started = time.perf_counter_ns()
                reopened = artifact.read_bytes()
                elapsed = time.perf_counter_ns() - started
                rows.append(_timing(run_id, image_id, preprocessing_id, codec_name, setting_id, "roundtrip", "read", iteration, elapsed, megapixels))
                started = time.perf_counter_ns()
                roundtrip_decoded = codec.decode(reopened)
                elapsed = time.perf_counter_ns() - started
                rows.append(_timing(run_id, image_id, preprocessing_id, codec_name, setting_id, "roundtrip", "decode", iteration, elapsed, megapixels))
                if roundtrip_decoded.shape != pixels.shape or roundtrip_decoded.dtype != pixels.dtype:
                    raise ValueError("Roundtrip decode changed image shape or dtype.")
            finally:
                artifact.unlink(missing_ok=True)
    assert encoded is not None
    return encoded, rows


def _timing(run_id, image_id, preprocessing_id, codec, setting_id, benchmark_kind, operation, iteration, elapsed, megapixels):
    return {"run_id": run_id, "image_id": image_id, "preprocessing_id": preprocessing_id, "codec": codec, "setting_id": setting_id, "benchmark_kind": benchmark_kind, "operation": operation, "iteration": iteration, "elapsed_ns": elapsed, "megapixels_per_second": megapixels / (elapsed / 1_000_000_000)}


def _add_segmentation(rows, run_id, image_id, preprocessing_id, codec, setting_id, method, reference, treatment, reference_pixels, treatment_pixels):
    threshold_for_crossing = reference.threshold if method == "fixed" else reference.threshold
    changed, fraction = threshold_crossings(reference_pixels, treatment_pixels, threshold_for_crossing)
    rows.append({"run_id": run_id, "image_id": image_id, "preprocessing_id": preprocessing_id, "codec": codec, "setting_id": setting_id, "segmentation_method": method, "threshold": treatment.threshold, "reference_threshold": reference.threshold, "treatment_threshold": treatment.threshold, "threshold_delta": treatment.threshold - reference.threshold, "object_count": len(treatment.components), "foreground_pixel_count": int(treatment.mask.sum()), "foreground_fraction": float(treatment.mask.mean()), "threshold_crossing_pixels": changed, "threshold_crossing_fraction": fraction, "engine": treatment.engine})


def _error(run_id, image_id, codec, setting_id, stage, exc):
    return {"run_id": run_id, "image_id": image_id, "codec": codec, "setting_id": setting_id, "stage": stage, "exception_type": type(exc).__name__, "message": str(exc)}


def _write_artifacts(output, image_id, setting_id, encoded, decoded):
    from PIL import Image
    directory = output / "artifacts" / image_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{setting_id}.bin").write_bytes(encoded)
    Image.fromarray(decoded, mode="L").save(directory / f"{setting_id}.decoded.png")

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CodecConfig:
    name: str
    qualities: list[int] | None = None
    options: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    id: str
    flatfield_enabled: bool
    flatfield_axis: int = 0
    flatfield_window_stride: int = 1
    flatfield_window_width: int = 1
    flatfield_min_field_value: float = 1.0
    flatfield_max_field_value: float | None = None
    pixel_metrics_enabled: bool = True
    segmentation_enabled: bool = True


@dataclass(frozen=True, slots=True)
class SystemBenchmarkConfig:
    enabled: bool = False
    directory: Path | None = None
    disk_file_sizes_mib: list[int] | None = None
    timed_runs: int = 3
    numpy_matrix_size: int = 512


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    source: Path
    raw: bytes
    dataset_path: Path
    dataset_type: str
    invert_grayscale: bool
    frame_selection: str
    frame_indices: list[int]
    methods: list[str]
    fixed_threshold: float
    minimum_area: int
    codecs: list[CodecConfig]
    preprocessing: list[PreprocessingConfig]
    system_benchmarks: SystemBenchmarkConfig
    warmup_runs: int
    timed_runs: int
    in_memory_enabled: bool
    roundtrip_enabled: bool
    roundtrip_directory: Path | None
    output_directory: Path
    retain_artifacts: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = source.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid TOML configuration: {exc}") from exc
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1.")
    root = source.parent
    dataset = _required_table(data, "dataset")
    invert_grayscale = dataset.get("invert_grayscale", False)
    if not isinstance(invert_grayscale, bool):
        raise ValueError("dataset.invert_grayscale must be true or false.")
    segmentation = _required_table(data, "segmentation")
    benchmark = _required_table(data, "benchmark")
    output = _required_table(data, "output")
    codecs_data = data.get("codecs")
    if not isinstance(codecs_data, list) or not codecs_data:
        raise ValueError("At least one [[codecs]] entry is required.")
    codecs: list[CodecConfig] = []
    for item in codecs_data:
        if not isinstance(item, dict):
            raise ValueError("Every [[codecs]] entry must be a TOML table.")
        qualities = item.get("qualities")
        if qualities is not None and (not isinstance(qualities, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in qualities)):
            raise ValueError("Codec qualities must be a list of integers.")
        codecs.append(CodecConfig(
            name=str(item.get("name", "")).lower(),
            qualities=qualities,
            options={key: value for key, value in item.items() if key not in {"name", "qualities"}},
        ))
    preprocessing_data = data.get("preprocessing", [{"id": "raw", "flatfield_enabled": False}])
    if not isinstance(preprocessing_data, list) or not preprocessing_data:
        raise ValueError("At least one [[preprocessing]] entry is required.")
    preprocessing = [PreprocessingConfig(
        id=str(item.get("id", "")),
        flatfield_enabled=bool(item.get("flatfield_enabled", False)),
        flatfield_axis=int(item.get("flatfield_axis", 0)),
        flatfield_window_stride=int(item.get("flatfield_window_stride", 1)),
        flatfield_window_width=int(item.get("flatfield_window_width", 1)),
        flatfield_min_field_value=float(item.get("flatfield_min_field_value", 1.0)),
        flatfield_max_field_value=(None if item.get("flatfield_max_field_value") is None else float(item["flatfield_max_field_value"])),
        pixel_metrics_enabled=bool(item.get("pixel_metrics_enabled", True)),
        segmentation_enabled=bool(item.get("segmentation_enabled", True)),
    ) for item in preprocessing_data]
    system_data = data.get("system_benchmarks", {})
    if not isinstance(system_data, dict):
        raise ValueError("[system_benchmarks] must be a TOML table.")
    system_directory_value = system_data.get("directory")
    system_benchmarks = SystemBenchmarkConfig(
        enabled=bool(system_data.get("enabled", False)),
        directory=(None if system_directory_value is None else ((root / str(system_directory_value)).resolve() if not Path(str(system_directory_value)).is_absolute() else Path(str(system_directory_value)))),
        disk_file_sizes_mib=[int(value) for value in system_data.get("disk_file_sizes_mib", [1, 16, 64])],
        timed_runs=int(system_data.get("timed_runs", 3)),
        numpy_matrix_size=int(system_data.get("numpy_matrix_size", 512)),
    )
    config = ExperimentConfig(
        source=source,
        raw=raw,
        dataset_path=(root / str(dataset.get("path", ""))).resolve() if not Path(str(dataset.get("path", ""))).is_absolute() else Path(str(dataset["path"])),
        dataset_type=str(dataset.get("type", "images")).lower(),
        invert_grayscale=invert_grayscale,
        frame_selection=str(dataset.get("frame_selection", "indices" if dataset.get("frame_indices") else "all")).lower(),
        frame_indices=[int(value) for value in dataset.get("frame_indices", [])],
        methods=[str(item).lower() for item in segmentation.get("methods", [])],
        fixed_threshold=float(segmentation.get("fixed_threshold", -1)),
        minimum_area=int(segmentation.get("minimum_area", 0)),
        codecs=codecs,
        preprocessing=preprocessing,
        system_benchmarks=system_benchmarks,
        warmup_runs=int(benchmark.get("warmup_runs", -1)),
        timed_runs=int(benchmark.get("timed_runs", -1)),
        in_memory_enabled=bool(benchmark.get("in_memory_enabled", True)),
        roundtrip_enabled=bool(benchmark.get("roundtrip_enabled", False)),
        roundtrip_directory=(None if benchmark.get("roundtrip_directory") is None else ((root / str(benchmark["roundtrip_directory"])).resolve() if not Path(str(benchmark["roundtrip_directory"])).is_absolute() else Path(str(benchmark["roundtrip_directory"])))),
        output_directory=(root / str(output.get("directory", ""))).resolve() if not Path(str(output.get("directory", ""))).is_absolute() else Path(str(output["directory"])),
        retain_artifacts=bool(output.get("retain_artifacts", False)),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if not config.dataset_path.is_dir():
        raise ValueError(f"dataset.path is not a directory: {config.dataset_path}")
    if config.dataset_type not in {"images", "video"}:
        raise ValueError("dataset.type must be 'images' or 'video'.")
    if config.frame_selection not in {"all", "indices"}:
        raise ValueError("dataset.frame_selection must be 'all' or 'indices'.")
    if config.dataset_type == "video" and config.frame_selection == "indices" and (not config.frame_indices or any(index < 0 for index in config.frame_indices)):
        raise ValueError("Video datasets using frame_selection='indices' require unique non-negative dataset.frame_indices.")
    if len(set(config.frame_indices)) != len(config.frame_indices):
        raise ValueError("dataset.frame_indices must be unique.")
    if any(not item.id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in item.id) for item in config.preprocessing):
        raise ValueError("preprocessing IDs must be non-empty lowercase letters, numbers, '_' or '-'.")
    if len({item.id for item in config.preprocessing}) != len(config.preprocessing):
        raise ValueError("preprocessing IDs must be unique.")
    for item in config.preprocessing:
        if item.flatfield_axis not in {0, 1} or item.flatfield_min_field_value < 0 or item.flatfield_window_stride < 1 or item.flatfield_window_width < 1 or item.flatfield_window_stride % 2 == 0 or item.flatfield_window_width % 2 == 0:
            raise ValueError("Invalid Pelagia flatfield settings.")
        if item.flatfield_max_field_value is not None and item.flatfield_max_field_value < item.flatfield_min_field_value:
            raise ValueError("flatfield_max_field_value must be >= flatfield_min_field_value.")
    if not config.methods or any(method not in {"fixed", "otsu"} for method in config.methods):
        raise ValueError("segmentation.methods must contain one or both of: fixed, otsu.")
    if not 0 <= config.fixed_threshold <= 255:
        raise ValueError("segmentation.fixed_threshold must be in [0, 255].")
    if config.minimum_area < 1:
        raise ValueError("segmentation.minimum_area must be at least 1.")
    if config.warmup_runs < 0 or config.timed_runs < 1:
        raise ValueError("benchmark.warmup_runs must be >= 0 and timed_runs must be >= 1.")
    if not config.in_memory_enabled and not config.roundtrip_enabled:
        raise ValueError("At least one benchmark mode must be enabled.")
    if config.roundtrip_enabled and config.roundtrip_directory is None:
        raise ValueError("roundtrip_enabled requires benchmark.roundtrip_directory.")
    system = config.system_benchmarks
    if system.enabled and system.directory is None:
        raise ValueError("Enabled system_benchmarks require system_benchmarks.directory.")
    if system.timed_runs < 1 or system.numpy_matrix_size < 16 or not system.disk_file_sizes_mib or any(size < 1 for size in system.disk_file_sizes_mib):
        raise ValueError("Invalid system benchmark settings.")
    output_root = config.output_directory.resolve()
    if config.roundtrip_enabled and config.roundtrip_directory is not None and config.roundtrip_directory.resolve() == output_root:
        raise ValueError("benchmark.roundtrip_directory must not equal output.directory.")
    if system.enabled and system.directory is not None and system.directory.resolve() == output_root:
        raise ValueError("system_benchmarks.directory must not equal output.directory.")
    seen: set[str] = set()
    from .codecs import codec_for

    for codec in config.codecs:
        if codec.name in seen:
            raise ValueError(f"Codec {codec.name!r} is repeated.")
        seen.add(codec.name)
        try:
            codec_for(codec.name).treatments(codec.qualities, codec.options)
        except ValueError as exc:
            raise ValueError(f"Invalid configuration for codec {codec.name!r}: {exc}") from exc


def _required_table(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] table is required.")
    return value

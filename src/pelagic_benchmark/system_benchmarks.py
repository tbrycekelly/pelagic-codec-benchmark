from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .config import SystemBenchmarkConfig


def run_system_benchmarks(config: SystemBenchmarkConfig, run_id: str) -> list[dict[str, object]]:
    """Run small contextual disk and NumPy benchmarks; never retain test files."""
    if not config.enabled:
        return []
    assert config.directory is not None
    directory = config.directory / run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        return _disk_rows(config, run_id, directory) + _numpy_rows(config, run_id)
    finally:
        for path in directory.glob("*.bin"):
            path.unlink(missing_ok=True)
        directory.rmdir()


def _disk_rows(config: SystemBenchmarkConfig, run_id: str, directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    block = np.random.default_rng(20260928).integers(0, 256, size=1024 * 1024, dtype=np.uint8).tobytes()
    for size_mib in config.disk_file_sizes_mib or []:
        size_bytes = size_mib * 1024 * 1024
        for iteration in range(config.timed_runs):
            path = directory / f"disk_{size_mib}mib_{iteration}.bin"
            try:
                started = time.perf_counter_ns()
                with path.open("wb", buffering=0) as stream:
                    remaining = size_bytes
                    while remaining:
                        chunk = block if remaining >= len(block) else block[:remaining]
                        stream.write(chunk)
                        remaining -= len(chunk)
                elapsed = time.perf_counter_ns() - started
                rows.append(_row(run_id, "disk", "write", f"{size_mib}MiB", iteration, size_bytes, elapsed, "page_cache_warm_after_write"))
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    started = time.perf_counter_ns()
                    os.fsync(descriptor)
                    elapsed = time.perf_counter_ns() - started
                finally:
                    os.close(descriptor)
                rows.append(_row(run_id, "disk", "fsync", f"{size_mib}MiB", iteration, size_bytes, elapsed, "durability_request"))
                started = time.perf_counter_ns()
                with path.open("rb", buffering=0) as stream:
                    while stream.read(len(block)):
                        pass
                elapsed = time.perf_counter_ns() - started
                rows.append(_row(run_id, "disk", "read", f"{size_mib}MiB", iteration, size_bytes, elapsed, "page_cache_warm_after_write"))
            finally:
                path.unlink(missing_ok=True)
    return rows


def _numpy_rows(config: SystemBenchmarkConfig, run_id: str) -> list[dict[str, object]]:
    size = config.numpy_matrix_size
    rng = np.random.default_rng(20260928)
    left = rng.random((size, size), dtype=np.float32)
    right = rng.random((size, size), dtype=np.float32)
    np.matmul(left, right)
    rows: list[dict[str, object]] = []
    for iteration in range(config.timed_runs):
        started = time.perf_counter_ns()
        np.matmul(left, right)
        elapsed = time.perf_counter_ns() - started
        rows.append(_row(run_id, "numpy", "matmul_float32", f"{size}x{size}", iteration, left.nbytes + right.nbytes, elapsed, "in_memory"))
        started = time.perf_counter_ns()
        np.fft.rfft2(left)
        elapsed = time.perf_counter_ns() - started
        rows.append(_row(run_id, "numpy", "rfft2_float32", f"{size}x{size}", iteration, left.nbytes, elapsed, "in_memory"))
    return rows


def _row(run_id: str, category: str, operation: str, setting: str, iteration: int, byte_count: int, elapsed_ns: int, cache_state: str) -> dict[str, object]:
    return {"run_id": run_id, "category": category, "operation": operation, "setting": setting, "iteration": iteration, "byte_count": byte_count, "elapsed_ns": elapsed_ns, "throughput_mib_per_second": byte_count / (1024 * 1024) / (elapsed_ns / 1_000_000_000), "cache_state": cache_state}

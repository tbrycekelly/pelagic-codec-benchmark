from __future__ import annotations

import math

import numpy as np


def pixel_metrics(reference: np.ndarray, treatment: np.ndarray) -> dict[str, float | int | bool | None]:
    if reference.shape != treatment.shape or reference.dtype != treatment.dtype:
        raise ValueError("Reference and treatment pixels must have equal shape and dtype.")
    diff = reference.astype(np.int16) - treatment.astype(np.int16)
    absolute = np.abs(diff)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "max_absolute_error": int(np.max(absolute)),
        "psnr": None if rmse == 0 else float(20 * math.log10(255.0 / rmse)),
        "exact_equal": bool(np.array_equal(reference, treatment)),
    }


def threshold_crossings(reference: np.ndarray, treatment: np.ndarray, threshold: float) -> tuple[int, float]:
    # This preserves the study's specified dark-side comparison verbatim.
    # Segmentation polarity is separately recorded in the manifest.
    changed = (reference < threshold) != (treatment < threshold)
    count = int(np.count_nonzero(changed))
    return count, count / reference.size

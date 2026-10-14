from __future__ import annotations

import numpy as np

from ..config import PreprocessingConfig
from ..dataset import pixel_hash


def apply_preprocessing(pixels: np.ndarray, condition: PreprocessingConfig, invert_grayscale: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    """Apply one fully specified, deterministic preprocessing condition."""
    if not condition.flatfield_enabled:
        result = np.ascontiguousarray(pixels)
        engine = "none"
    else:
        from Pelagia.processing.frame_correction import flatfield_profile_correction

        if condition.flatfield_window_stride != 1 or condition.flatfield_window_width != 1:
            raise ValueError("Temporal Pelagia flatfield windows require the profile-generation pass; this vertical slice currently supports Pelagia's explicit per-frame window (stride=1, width=1).")
        # This is Pelagia's profile representation for a one-frame window:
        # sum over the stack axis, divided by its value count.
        profile = np.mean(pixels, axis=condition.flatfield_axis, dtype=np.float64).astype(np.float32)
        result = flatfield_profile_correction(
            pixels,
            profile,
            axis=condition.flatfield_axis,
            min_field_value=condition.flatfield_min_field_value,
            max_field_value=condition.flatfield_max_field_value,
        )
        engine = "Pelagia.processing.frame_correction.flatfield_profile_correction"
    if result.dtype != np.uint8 or result.shape != pixels.shape:
        raise ValueError("Preprocessing did not produce same-shape uint8 pixels.")
    if invert_grayscale:
        result = np.ascontiguousarray(np.bitwise_not(result))
    return result, {"preprocessing_id": condition.id, "flatfield_enabled": condition.flatfield_enabled, "flatfield_method": ("window_column_mean" if condition.flatfield_axis == 0 else "window_row_mean") if condition.flatfield_enabled else None, "flatfield_axis": condition.flatfield_axis if condition.flatfield_enabled else None, "flatfield_window_stride": condition.flatfield_window_stride if condition.flatfield_enabled else None, "flatfield_window_width": condition.flatfield_window_width if condition.flatfield_enabled else None, "flatfield_min_field_value": condition.flatfield_min_field_value if condition.flatfield_enabled else None, "flatfield_max_field_value": condition.flatfield_max_field_value if condition.flatfield_enabled else None, "canonical_inversion": invert_grayscale, "inversion_stage": "after_preprocessing" if invert_grayscale else None, "engine": engine, "pixel_sha256": pixel_hash(result)}

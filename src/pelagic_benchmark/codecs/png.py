from __future__ import annotations

from collections.abc import Mapping

import imagecodecs
import numpy as np

from .base import Treatment


class PngCodec:
    name = "png"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        levels = _integer_values(options, "deflate_levels", [6], 0, 9)
        return [Treatment("png", f"png_deflate_l{level}", {"deflate_level": level}, True) for level in levels]

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.png_encode(image, level=int(settings.get("deflate_level", 6))))

    def decode(self, data: bytes) -> np.ndarray:
        result = imagecodecs.png_decode(data)
        if result.dtype != np.uint8 or result.ndim != 2:
            raise ValueError("PNG codec did not decode to 2D uint8 grayscale pixels.")
        return np.ascontiguousarray(result)

    def implementation_metadata(self) -> dict[str, str]:
        return _metadata("libpng", "PNG / DEFLATE")


def _integer_values(options: Mapping[str, object] | None, key: str, default: list[int], minimum: int, maximum: int) -> list[int]:
    value = default if options is None or key not in options else options[key]
    if not isinstance(value, list) or not value or any(isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum for item in value):
        raise ValueError(f"{key} must be a non-empty list of integers in [{minimum}, {maximum}].")
    if len(set(value)) != len(value):
        raise ValueError(f"{key} must not contain duplicates.")
    return value


def _metadata(library: str, format_name: str) -> dict[str, str]:
    return {"library": library, "binding": "imagecodecs", "binding_version": imagecodecs.__version__, "format": format_name}

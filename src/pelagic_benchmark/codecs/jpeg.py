from __future__ import annotations

from collections.abc import Mapping

import imagecodecs
import numpy as np

from .base import Treatment
from .png import _metadata


class JpegCodec:
    name = "jpeg"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        if not values or any(isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100 for quality in values):
            raise ValueError("qualities must be a non-empty list of integers in [1, 100].")
        if len(set(values)) != len(values):
            raise ValueError("qualities must not contain duplicates.")
        return [
            Treatment("jpeg", f"jpeg_q{quality}", {"quality": int(quality)}, False)
            for quality in values
        ]

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.jpeg_encode(image, level=int(settings["quality"])))

    def decode(self, data: bytes) -> np.ndarray:
        result = imagecodecs.jpeg_decode(data)
        if result.dtype != np.uint8 or result.ndim != 2:
            raise ValueError("JPEG codec did not decode to 2D uint8 grayscale pixels.")
        return np.ascontiguousarray(result)

    def implementation_metadata(self) -> dict[str, str]:
        return _metadata("libjpeg-turbo", "JPEG")

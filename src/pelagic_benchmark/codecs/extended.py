"""Native codec adapters for the formats beyond PNG and baseline JPEG.

The benchmark's scientific input is always a two-dimensional uint8 grayscale
array. WebP requires RGB input. AVIF uses the same explicit RGB adapter so
quality-level treatments follow one documented path. Both adapters repeat the
gray channel for encoding and select the first decoded channel for comparison.
This conversion is part of those treatments' recorded settings.
"""

from __future__ import annotations

import io
from collections.abc import Mapping

import imagecodecs
import numpy as np
import tifffile

from .base import Treatment
from .png import _integer_values, _metadata


def _no_unknown_options(options: Mapping[str, object] | None, allowed: set[str]) -> Mapping[str, object]:
    result = options or {}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"unsupported setting(s): {', '.join(sorted(unknown))}.")
    return result


def _optional_bool(options: Mapping[str, object], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false.")
    return value


def _gray(image: np.ndarray, format_name: str) -> np.ndarray:
    result = np.asarray(image)
    if result.dtype != np.uint8 or result.ndim != 2:
        raise ValueError(f"{format_name} codec did not decode to 2D uint8 grayscale pixels.")
    return np.ascontiguousarray(result)


def _rgb_gray(image: np.ndarray) -> np.ndarray:
    return np.repeat(np.ascontiguousarray(image)[..., np.newaxis], 3, axis=2)


def _decoded_rgb_gray(image: np.ndarray, format_name: str) -> np.ndarray:
    result = np.asarray(image)
    if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] < 3:
        raise ValueError(f"{format_name} codec did not decode to 8-bit RGB pixels.")
    return np.ascontiguousarray(result[..., 0])


class TiffCodec:
    name = "tiff"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"compressions", "compression_level"})
        compressions = options.get("compressions", ["none", "deflate", "zstd"])
        if not isinstance(compressions, list) or not compressions:
            raise ValueError("compressions must be a non-empty list of: none, deflate, zstd.")
        allowed = {"none", "deflate", "zstd"}
        if any(not isinstance(value, str) or value.lower() not in allowed for value in compressions):
            raise ValueError("compressions must be a non-empty list of: none, deflate, zstd.")
        normalized = [value.lower() for value in compressions]
        if len(set(normalized)) != len(normalized):
            raise ValueError("compressions must not contain duplicates.")
        level = options.get("compression_level", 6)
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 22:
            raise ValueError("compression_level must be an integer in [0, 22].")
        if "deflate" in normalized and level > 9:
            raise ValueError("compression_level must be at most 9 when Deflate is selected.")
        treatments = []
        for compression in normalized:
            settings: dict[str, object] = {"compression": compression}
            if compression != "none":
                settings["compression_level"] = level
            treatments.append(Treatment("tiff", f"tiff_{compression}" + (f"_l{level}" if compression != "none" else ""), settings, True))
        return treatments

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        compression = settings["compression"]
        target = io.BytesIO()
        kwargs: dict[str, object] = {"photometric": "minisblack", "metadata": None, "compression": None if compression == "none" else compression}
        if compression != "none":
            kwargs["compressionargs"] = {"level": int(settings["compression_level"])}
        tifffile.imwrite(target, image, **kwargs)
        return target.getvalue()

    def decode(self, data: bytes) -> np.ndarray:
        return _gray(tifffile.imread(io.BytesIO(data)), "TIFF")

    def implementation_metadata(self) -> dict[str, str]:
        return {"library": "libtiff + libdeflate/Zstandard", "binding": "tifffile + imagecodecs", "tifffile_version": tifffile.__version__, "imagecodecs_version": imagecodecs.__version__, "format": "TIFF"}


class Jpeg2000Codec:
    name = "jpeg2000"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"lossless", "lossy_levels"})
        lossless = _optional_bool(options, "lossless", True)
        levels = _integer_values(options, "lossy_levels", [], 1, 100) if "lossy_levels" in options else []
        if not lossless and not levels:
            raise ValueError("set lossless = true and/or provide lossy_levels.")
        treatments = [Treatment("jpeg2000", "jpeg2000_lossless", {"reversible": True, "level": 0}, True)] if lossless else []
        treatments.extend(Treatment("jpeg2000", f"jpeg2000_l{level}", {"reversible": False, "level": level}, False) for level in levels)
        return treatments

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.jpeg2k_encode(image, level=int(settings["level"]), reversible=bool(settings["reversible"])))

    def decode(self, data: bytes) -> np.ndarray:
        return _gray(imagecodecs.jpeg2k_decode(data), "JPEG 2000")

    def implementation_metadata(self) -> dict[str, str]:
        return _metadata("OpenJPEG", "JPEG 2000")


class JpegXlCodec:
    name = "jpegxl"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"lossless", "distances", "effort"})
        lossless = _optional_bool(options, "lossless", True)
        distances = options.get("distances", [])
        if not isinstance(distances, list) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in distances):
            raise ValueError("distances must be a list of positive numbers.")
        if len(set(distances)) != len(distances):
            raise ValueError("distances must not contain duplicates.")
        effort = options.get("effort", 7)
        if isinstance(effort, bool) or not isinstance(effort, int) or not 1 <= effort <= 10:
            raise ValueError("effort must be an integer in [1, 10].")
        if not lossless and not distances:
            raise ValueError("set lossless = true and/or provide distances.")
        treatments = [Treatment("jpegxl", "jpegxl_lossless", {"lossless": True, "effort": effort}, True)] if lossless else []
        treatments.extend(Treatment("jpegxl", f"jpegxl_d{_number_id(distance)}", {"lossless": False, "distance": float(distance), "effort": effort}, False) for distance in distances)
        return treatments

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.jpegxl_encode(image, **settings))

    def decode(self, data: bytes) -> np.ndarray:
        return _gray(imagecodecs.jpegxl_decode(data), "JPEG XL")

    def implementation_metadata(self) -> dict[str, str]:
        return _metadata("libjxl", "JPEG XL")


class WebpCodec:
    name = "webp"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"lossless", "method"})
        lossless = _optional_bool(options, "lossless", True)
        qualities = values or []
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100 for value in qualities) or len(set(qualities)) != len(qualities):
            raise ValueError("qualities must be unique integers in [0, 100].")
        if not lossless and not qualities:
            raise ValueError("set lossless = true and/or provide qualities.")
        method = options.get("method", 4)
        if isinstance(method, bool) or not isinstance(method, int) or not 0 <= method <= 6:
            raise ValueError("method must be an integer in [0, 6].")
        treatments = [Treatment("webp", "webp_lossless", {"lossless": True, "method": method, "gray_adapter": "rgb_repeat_first_channel"}, True)] if lossless else []
        treatments.extend(Treatment("webp", f"webp_q{quality}", {"lossless": False, "quality": quality, "method": method, "gray_adapter": "rgb_repeat_first_channel"}, False) for quality in qualities)
        return treatments

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.webp_encode(_rgb_gray(image), level=int(settings["quality"]) if "quality" in settings else None, lossless=bool(settings["lossless"]), method=int(settings["method"])))

    def decode(self, data: bytes) -> np.ndarray:
        return _decoded_rgb_gray(imagecodecs.webp_decode(data), "WebP")

    def implementation_metadata(self) -> dict[str, str]:
        metadata = _metadata("libwebp", "WebP")
        metadata["grayscale_adapter"] = "RGB repetition before encode; first decoded RGB channel compared"
        return metadata


class AvifCodec:
    name = "avif"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"quality_levels", "speed"})
        levels = _integer_values(options, "quality_levels", [20, 30, 40, 50], 0, 63)
        speed = options.get("speed", 6)
        if isinstance(speed, bool) or not isinstance(speed, int) or not 0 <= speed <= 10:
            raise ValueError("speed must be an integer in [0, 10].")
        return [Treatment("avif", f"avif_q{level}", {"quality_level": level, "speed": speed, "gray_adapter": "rgb_repeat_first_channel"}, False) for level in levels]

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.avif_encode(_rgb_gray(image), level=int(settings["quality_level"]), speed=int(settings["speed"]), pixelformat="yuv444"))

    def decode(self, data: bytes) -> np.ndarray:
        return _decoded_rgb_gray(imagecodecs.avif_decode(data), "AVIF")

    def implementation_metadata(self) -> dict[str, str]:
        metadata = _metadata("libavif", "AVIF")
        metadata["grayscale_adapter"] = "RGB repetition before encode; YUV444; first decoded RGB channel compared"
        metadata["quality_scale"] = "imagecodecs/libavif level 0 (lowest) to 63 (highest)"
        return metadata


class JpegLsCodec:
    name = "jpegls"

    def treatments(self, values: list[int] | None = None, options: Mapping[str, object] | None = None) -> list[Treatment]:
        options = _no_unknown_options(options, {"lossless", "near_lossless"})
        lossless = _optional_bool(options, "lossless", True)
        near = _integer_values(options, "near_lossless", [], 1, 255) if "near_lossless" in options else []
        if not lossless and not near:
            raise ValueError("set lossless = true and/or provide near_lossless.")
        treatments = [Treatment("jpegls", "jpegls_lossless", {"near": 0}, True)] if lossless else []
        treatments.extend(Treatment("jpegls", f"jpegls_near{level}", {"near": level}, False) for level in near)
        return treatments

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes:
        return bytes(imagecodecs.jpegls_encode(image, level=int(settings["near"])))

    def decode(self, data: bytes) -> np.ndarray:
        return _gray(imagecodecs.jpegls_decode(data), "JPEG-LS")

    def implementation_metadata(self) -> dict[str, str]:
        return _metadata("CharLS", "JPEG-LS")


def _number_id(value: int | float) -> str:
    return str(value).replace(".", "p")

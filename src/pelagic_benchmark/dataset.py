from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import cv2
from PIL import Image


SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class CanonicalImage:
    image_id: str
    source_path: str
    source_sha256: str
    pixel_sha256: str
    source_frame_index: int | None
    pixels: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(pixels: np.ndarray) -> str:
    image = np.ascontiguousarray(pixels)
    metadata = json.dumps({"dtype": image.dtype.str, "shape": image.shape}, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(b"pelagic-benchmark/canonical/v1\0" + metadata + b"\0" + image.tobytes()).hexdigest()


def load_canonical(path: Path, dataset_root: Path) -> CanonicalImage:
    try:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected 2D grayscale uint8 image (Pillow mode L), got mode {image.mode!r}.")
            pixels = np.ascontiguousarray(np.asarray(image))
    except OSError as exc:
        raise ValueError(f"Could not decode image {path}: {exc}") from exc
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 image, got shape={pixels.shape}, dtype={pixels.dtype}.")
    source_sha = sha256_file(path)
    relative = path.relative_to(dataset_root).as_posix()
    return CanonicalImage(
        image_id=f"img_{source_sha[:16]}",
        source_path=relative,
        source_sha256=source_sha,
        pixel_sha256=pixel_hash(pixels),
        source_frame_index=None,
        pixels=pixels,
    )


def discover_images(dataset_root: Path) -> list[Path]:
    paths = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported image files found under {dataset_root}.")
    return paths


def discover_videos(dataset_root: Path) -> list[Path]:
    paths = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"})
    if not paths:
        raise ValueError(f"No supported video files found under {dataset_root}.")
    return paths


def estimate_video_frame_count(paths: list[Path]) -> int:
    """Return container-reported frame counts without decoding the streams."""
    total = 0
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        try:
            total += max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
    return total


def iter_video_frames(path: Path, dataset_root: Path, frame_selection: str, frame_indices: list[int]) -> Iterator[CanonicalImage]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {path}.")
    source_sha = sha256_file(path)
    relative = path.relative_to(dataset_root).as_posix()
    requested = set(frame_indices)
    found: set[int] = set()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_selection == "all" or frame_index in requested:
                yield _canonical_video_frame(frame, source_sha, relative, frame_index)
                found.add(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if frame_selection == "indices" and requested != found:
        raise ValueError(f"Requested frame indices {sorted(requested - found)} could not be decoded from {path}.")


def _canonical_video_frame(frame: np.ndarray, source_sha: str, relative: str, frame_index: int) -> CanonicalImage:
    if frame.ndim == 2:
        pixels = frame
    elif frame.ndim == 3 and frame.shape[2] == 3 and np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2]):
        pixels = frame[..., 0]
    else:
        raise ValueError(f"Video frame {frame_index} is color; only unambiguous grayscale video is supported.")
    pixels = np.ascontiguousarray(pixels)
    if pixels.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video pixels, got {pixels.dtype}.")
    return CanonicalImage(
        image_id=f"img_{source_sha[:16]}_f{frame_index:06d}",
        source_path=relative,
        source_sha256=source_sha,
        pixel_sha256=pixel_hash(pixels),
        source_frame_index=frame_index,
        pixels=pixels,
    )


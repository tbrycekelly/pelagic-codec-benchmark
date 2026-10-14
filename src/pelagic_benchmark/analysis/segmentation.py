from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Component:
    component_id: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    perimeter: float


@dataclass(frozen=True, slots=True)
class Segmentation:
    mask: np.ndarray
    labels: np.ndarray
    components: list[Component]
    threshold: float
    engine: str


def pelagia_metadata() -> dict[str, str | None]:
    try:
        from importlib.metadata import version
        import Pelagia.processing.frame_threshold  # noqa: F401

        return {"version": version("pelagia"), "integration": "Pelagia.processing.frame_threshold"}
    except Exception:
        return {"version": None, "integration": "opencv_fallback"}


def otsu_threshold(image: np.ndarray) -> tuple[float, str]:
    try:
        from Pelagia.processing.frame_threshold import otsu_threshold_value

        return float(otsu_threshold_value(image)), "pelagia"
    except Exception:
        value, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return float(value), "opencv_fallback"


def segment(image: np.ndarray, method: str, fixed_threshold: float, minimum_area: int) -> Segmentation:
    if method == "fixed":
        threshold, engine = fixed_threshold, "pelagia" if _pelagia_threshold_available() else "opencv_fallback"
    elif method == "otsu":
        threshold, engine = otsu_threshold(image)
    else:
        raise ValueError(f"Unsupported segmentation method {method!r}.")
    mask = _manual_mask(image, threshold, engine)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Component labels can exceed 255 on noisy/strongly compressed frames.
    # Keep a wide label image; only the final binary mask is uint8-equivalent.
    retained = np.zeros(mask.shape, dtype=np.int32)
    components: list[Component] = []
    next_id = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        retained[labels == label] = next_id
        x, y, width, height = (int(stats[label, index]) for index in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        local = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
        components.append(Component(next_id, area, float(centroids[label][0]), float(centroids[label][1]), x, y, width, height, perimeter))
        next_id += 1
    return Segmentation(retained > 0, retained, components, float(threshold), engine)


def _pelagia_threshold_available() -> bool:
    try:
        from Pelagia.processing.frame_threshold import threshold_manual  # noqa: F401
        return True
    except Exception:
        return False


def _manual_mask(image: np.ndarray, threshold: float, engine: str) -> np.ndarray:
    if engine == "pelagia":
        from Pelagia.processing.frame_threshold import threshold_manual

        return np.ascontiguousarray(threshold_manual(image, threshold) > 0).astype(np.uint8)
    _, mask = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return np.ascontiguousarray(mask)

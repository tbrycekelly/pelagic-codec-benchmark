from __future__ import annotations

import math

import numpy as np

from .segmentation import Segmentation


def match_components(reference: Segmentation, treatment: Segmentation) -> list[dict[str, object]]:
    ref_by_id = {item.component_id: item for item in reference.components}
    treat_by_id = {item.component_id: item for item in treatment.components}
    # Count label-pair co-occurrences in one pass over the full frame. This is
    # mathematically identical to pairwise mask intersections, but avoids an
    # O(reference_components * treatment_components * pixels) scan on noisy
    # low-quality decodes.
    label_width = int(treatment.labels.max()) + 1
    overlap_pixels = (reference.labels > 0) & (treatment.labels > 0)
    encoded_pairs = reference.labels[overlap_pixels].astype("int64") * label_width + treatment.labels[overlap_pixels]
    pair_codes, counts = np.unique(encoded_pairs, return_counts=True)
    overlaps = [(int(code // label_width), int(code % label_width), int(count)) for code, count in zip(pair_codes, counts, strict=True)]
    ref_degree = {component_id: 0 for component_id in ref_by_id}
    treatment_degree = {component_id: 0 for component_id in treat_by_id}
    for ref_id, treatment_id, _ in overlaps:
        ref_degree[ref_id] += 1
        treatment_degree[treatment_id] += 1
    rows: list[dict[str, object]] = []
    for ref_id, treatment_id, intersection in overlaps:
        ref = ref_by_id[ref_id]
        treat = treat_by_id[treatment_id]
        union = ref.area + treat.area - intersection
        if ref_degree[ref_id] > 1 and treatment_degree[treatment_id] > 1:
            relation = "split_merge"
        elif ref_degree[ref_id] > 1:
            relation = "split"
        elif treatment_degree[treatment_id] > 1:
            relation = "merge"
        else:
            relation = "1:1"
        rows.append(_row(relation, ref, treat, intersection, union))
    for ref_id, ref in ref_by_id.items():
        if not ref_degree[ref_id]:
            rows.append(_row("lost", ref, None, 0, ref.area))
    for treatment_id, treat in treat_by_id.items():
        if not treatment_degree[treatment_id]:
            rows.append(_row("new", None, treat, 0, treat.area))
    return rows


def _row(relation: str, reference, treatment, intersection: int, union: int) -> dict[str, object]:
    reference_area = reference.area if reference else None
    treatment_area = treatment.area if treatment else None
    relative_error = None if reference_area in (None, 0) or treatment_area is None else (treatment_area - reference_area) / reference_area
    dx = dy = distance = None
    if reference and treatment:
        dx = treatment.centroid_x - reference.centroid_x
        dy = treatment.centroid_y - reference.centroid_y
        distance = math.hypot(dx, dy)
    return {
        "relationship": relation,
        "reference_component_id": reference.component_id if reference else None,
        "treatment_component_id": treatment.component_id if treatment else None,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else None,
        "reference_area": reference_area,
        "treatment_area": treatment_area,
        "relative_area_error": relative_error,
        "centroid_dx": dx,
        "centroid_dy": dy,
        "centroid_displacement": distance,
        "reference_perimeter": reference.perimeter if reference else None,
        "treatment_perimeter": treatment.perimeter if treatment else None,
        "reference_bbox": _bbox(reference),
        "treatment_bbox": _bbox(treatment),
    }


def _bbox(component) -> str | None:
    if component is None:
        return None
    return f"{component.bbox_x},{component.bbox_y},{component.bbox_width},{component.bbox_height}"

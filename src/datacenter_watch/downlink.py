"""Shared packet, hash, and diff logic for the satellite/ground pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from datacenter_watch.compact_schema import (
    DETECTION_EVAL_FIELDS,
    TILE_CONTEXT_FIELDS as TILE_CONTEXT_EVAL_FIELDS,
    is_positive_detection,
)

BBOX_ROUND_DIGITS = 4
BBOX_EPSILON = 1e-3


def _rounded_bbox(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    rounded: list[float] = []
    for raw in value:
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = 0.0
        rounded.append(round(min(1.0, max(0.0, number)), BBOX_ROUND_DIGITS))
    return rounded


def bbox_centroid(bbox: object) -> tuple[float, float]:
    x1, y1, x2, y2 = _rounded_bbox(bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def normalize_detection(detection: object) -> dict[str, object]:
    source = detection if isinstance(detection, dict) else {}
    normalized: dict[str, object] = {"bbox": _rounded_bbox(source.get("bbox"))}
    for field in DETECTION_EVAL_FIELDS:
        normalized[field] = source.get(field)
    return normalized


def canonical_payload(payload: dict[str, object]) -> dict[str, object]:
    detections_raw = payload.get("detections")
    detections = []
    if isinstance(detections_raw, list):
        detections = [
            normalize_detection(detection)
            for detection in detections_raw
            if is_positive_detection(detection)
        ]
    detections.sort(
        key=lambda detection: (
            *bbox_centroid(detection.get("bbox")),
            str(detection.get("site_class", "")),
            str(detection.get("construction_stage", "")),
        )
    )

    tile_context_raw = payload.get("tile_context")
    tile_context_source = tile_context_raw if isinstance(tile_context_raw, dict) else {}
    tile_context = {
        field: tile_context_source.get(field)
        for field in TILE_CONTEXT_EVAL_FIELDS
    }
    return {
        "detections": detections,
        "tile_context": tile_context,
    }


def payload_hash(payload: dict[str, object]) -> str:
    canonical = canonical_payload(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def has_positive_detections(payload: dict[str, object]) -> bool:
    detections = payload.get("detections")
    return isinstance(detections, list) and any(is_positive_detection(detection) for detection in detections)


def _bbox_changed(current: object, previous: object) -> bool:
    current_bbox = _rounded_bbox(current)
    previous_bbox = _rounded_bbox(previous)
    return any(abs(a - b) > BBOX_EPSILON for a, b in zip(current_bbox, previous_bbox, strict=True))


def _detection_distance(current: dict[str, object], previous: dict[str, object]) -> float:
    current_x, current_y = bbox_centroid(current.get("bbox"))
    previous_x, previous_y = bbox_centroid(previous.get("bbox"))
    return math.hypot(current_x - previous_x, current_y - previous_y)


def _match_detection_indices(
    current: list[dict[str, object]],
    previous: list[dict[str, object]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for current_idx, current_detection in enumerate(current):
        for previous_idx, previous_detection in enumerate(previous):
            candidates.append(
                (_detection_distance(current_detection, previous_detection), current_idx, previous_idx)
            )
    candidates.sort()
    matched_current: set[int] = set()
    matched_previous: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, current_idx, previous_idx in candidates:
        if current_idx in matched_current or previous_idx in matched_previous:
            continue
        matched_current.add(current_idx)
        matched_previous.add(previous_idx)
        pairs.append((current_idx, previous_idx))
    return pairs


def field_level_diff(
    current_payload: dict[str, object],
    previous_payload: dict[str, object],
) -> dict[str, object]:
    current = canonical_payload(current_payload)
    previous = canonical_payload(previous_payload)

    current_detections = current["detections"]
    previous_detections = previous["detections"]
    assert isinstance(current_detections, list)
    assert isinstance(previous_detections, list)

    tile_context_fields = [
        field
        for field in TILE_CONTEXT_EVAL_FIELDS
        if current["tile_context"].get(field) != previous["tile_context"].get(field)  # type: ignore[union-attr]
    ]

    changes: list[dict[str, Any]] = []
    matched_current: set[int] = set()
    matched_previous: set[int] = set()
    for current_idx, previous_idx in _match_detection_indices(current_detections, previous_detections):
        matched_current.add(current_idx)
        matched_previous.add(previous_idx)
        current_detection = current_detections[current_idx]
        previous_detection = previous_detections[previous_idx]
        field_changes: list[str] = []
        if _bbox_changed(current_detection.get("bbox"), previous_detection.get("bbox")):
            field_changes.append("bbox")
        for field in DETECTION_EVAL_FIELDS:
            if current_detection.get(field) != previous_detection.get(field):
                field_changes.append(field)
        if field_changes:
            changes.append(
                {
                    "type": "changed",
                    "current": current_detection,
                    "previous": previous_detection,
                    "fields": field_changes,
                }
            )

    for current_idx, detection in enumerate(current_detections):
        if current_idx not in matched_current:
            changes.append({"type": "added", "current": detection})
    for previous_idx, detection in enumerate(previous_detections):
        if previous_idx not in matched_previous:
            changes.append({"type": "removed", "previous": detection})

    return {
        "tile_context_fields": tile_context_fields,
        "detections": changes,
    }


def has_meaningful_change(diff: dict[str, object]) -> bool:
    detections = diff.get("detections")
    tile_context_fields = diff.get("tile_context_fields")
    return bool(detections) or bool(tile_context_fields)


def packet_id(packet: dict[str, object]) -> str:
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def detection_centroid_lon_lat(
    tile_lon: float,
    tile_lat: float,
    size_km: float,
    bbox: object,
) -> tuple[float, float]:
    center_x, center_y = bbox_centroid(bbox)
    lat_span = size_km / 111.0
    lon_span = size_km / (111.0 * math.cos(math.radians(tile_lat)))
    lon = tile_lon + (center_x - 0.5) * lon_span
    lat = tile_lat - (center_y - 0.5) * lat_span
    return round(lon, 6), round(lat, 6)

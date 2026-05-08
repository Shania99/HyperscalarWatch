"""Shared compact annotation schema and migration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SITE_CLASS_ENUM = [
    "data_center",
    "industrial_site",
    "no_site",
]

CONSTRUCTION_STAGE_ENUM = [
    "undisturbed",
    "active_construction",
    "operational",
]

DETECTION_OUTPUT_FIELDS = [
    "site_class",
    "construction_stage",
    "roof_bright_membrane",
    "bare_soil_present",
    "reasoning",
]

DETECTION_EVAL_FIELDS = [
    "site_class",
    "construction_stage",
    "roof_bright_membrane",
    "bare_soil_present",
]

TILE_CONTEXT_FIELDS = [
    "image_quality_limited",
]

NON_POSITIVE_SITE_CLASSES = {"no_site"}

OLD_SITE_CLASS_MAP = {
    "data_center": "data_center",
    "under_construction": "industrial_site",
    "industrial_logistics_warehouse": "industrial_site",
    "industrial_manufacturing": "industrial_site",
    "industrial_semiconductor_fab": "industrial_site",
    "power_generation_facility": "industrial_site",
    "ambiguous_large_industrial": "industrial_site",
    "no_industrial_site_present": "no_site",
}

ACTIVE_OLD_STAGES = {
    "land_clearing",
    "earthworks",
    "foundations",
    "structural_shell",
    "roof_complete",
    "expansion",
}

RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            {"type": "null"},
                        ]
                    },
                    "site_class": {"type": "string", "enum": SITE_CLASS_ENUM},
                    "construction_stage": {
                        "type": "string",
                        "enum": CONSTRUCTION_STAGE_ENUM,
                    },
                    "roof_bright_membrane": {"type": "boolean"},
                    "bare_soil_present": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "bbox",
                    "site_class",
                    "construction_stage",
                    "roof_bright_membrane",
                    "bare_soil_present",
                    "reasoning",
                ],
            },
        },
        "tile_context": {
            "type": "object",
            "properties": {
                "image_quality_limited": {"type": "boolean"},
            },
            "required": ["image_quality_limited"],
        },
    },
    "required": ["detections", "tile_context"],
}


def is_positive_site_class(site_class: object) -> bool:
    return str(site_class) not in NON_POSITIVE_SITE_CLASSES


def is_positive_detection(detection: object) -> bool:
    return isinstance(detection, Mapping) and is_positive_site_class(detection.get("site_class"))


def _normalized_bbox(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    bbox: list[float] = []
    for raw in value:
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
        bbox.append(min(1.0, max(0.0, number)))
    return bbox


def map_site_class(old_site_class: object) -> str:
    old_value = str(old_site_class or "").strip()
    if old_value in SITE_CLASS_ENUM:
        return old_value
    return OLD_SITE_CLASS_MAP.get(old_value, "industrial_site")


def map_construction_stage(old_stage: object) -> str:
    stage = str(old_stage or "").strip()
    if stage in CONSTRUCTION_STAGE_ENUM:
        return stage
    if stage == "operational":
        return "operational"
    if stage == "undisturbed":
        return "undisturbed"
    if stage in ACTIVE_OLD_STAGES:
        return "active_construction"
    return "undisturbed"


def migrate_detection(source: Mapping[str, Any]) -> dict[str, object]:
    site_class = map_site_class(source.get("site_class"))
    bbox = _normalized_bbox(source.get("bbox"))
    return {
        "bbox": bbox,
        "site_class": site_class,
        "construction_stage": map_construction_stage(source.get("construction_stage")),
        "roof_bright_membrane": bool(source.get("roof_bright_membrane")),
        "bare_soil_present": bool(source.get("bare_soil_present")),
        "reasoning": str(source.get("reasoning") or "").strip(),
    }


def migrate_tile_context(source: object) -> dict[str, object]:
    context = source if isinstance(source, Mapping) else {}
    return {
        "image_quality_limited": bool(context.get("image_quality_limited")),
    }


def migrate_annotation(annotation: Mapping[str, Any]) -> dict[str, object]:
    migrated = dict(annotation)
    detections_raw = annotation.get("detections")
    detections: list[dict[str, object]] = []
    if isinstance(detections_raw, list):
        for detection in detections_raw:
            if isinstance(detection, Mapping):
                detections.append(migrate_detection(detection))
    migrated["detections"] = detections
    migrated["tile_context"] = migrate_tile_context(annotation.get("tile_context"))
    return migrated

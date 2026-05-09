"""Satellite-side change suppression and downlink packet formatting."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TypeAlias

from datacenter_watch.downlink import (
    canonical_payload,
    field_level_diff,
    has_meaningful_change,
    packet_id,
    payload_hash,
)

CacheEntry: TypeAlias = tuple[dict[str, object], str]

_store: dict[str, CacheEntry] = {}


def clear_cache() -> None:
    _store.clear()


def cache_size() -> int:
    return len(_store)


def _build_packet(
    *,
    tile_id: str,
    tile_lon: float,
    tile_lat: float,
    size_km: float,
    observed_at: str,
    payload: dict[str, object],
    payload_hash_value: str,
    change_type: str,
    image_bytes: dict[str, bytes | None] | None = None,
    diff: dict[str, object] | None = None,
) -> dict[str, object]:
    packet: dict[str, object] = {
        "tile_id": tile_id,
        "tile_center_lon": tile_lon,
        "tile_center_lat": tile_lat,
        "size_km": size_km,
        "observed_at": observed_at,
        "payload_hash": payload_hash_value[:16],
        "change_type": change_type,
        "detections": payload["detections"],
        "tile_context": payload["tile_context"],
    }
    if image_bytes is not None:
        packet["images"] = {
            name: {
                "mime_type": "image/png",
                "data_b64": base64.b64encode(raw).decode("ascii"),
            }
            for name, raw in image_bytes.items()
            if raw
        }
    if diff is not None and has_meaningful_change(diff):
        packet["diff"] = diff
    packet["packet_id"] = packet_id(packet)
    return packet


def observe_tile(
    *,
    tile_id: str,
    tile_lon: float,
    tile_lat: float,
    size_km: float,
    observed_at: str,
    payload: dict[str, object],
    image_bytes: dict[str, bytes | None] | None = None,
) -> dict[str, object]:
    canonical = canonical_payload(payload)
    current_hash = payload_hash(canonical)
    # All subtiles of the same location (e.g. s00, s01, s02) share one cache slot
    cache_key = tile_id.split("/")[0] if "/" in tile_id else tile_id
    cached = _store.get(cache_key)
    has_detections = bool(canonical["detections"])

    if not has_detections:
        if cached is None:
            return {"transmitted": False, "reason": "empty"}
        previous_payload, _previous_hash = cached
        diff = field_level_diff(canonical, previous_payload)
        _store.pop(cache_key, None)
        packet = _build_packet(
            tile_id=tile_id,
            tile_lon=tile_lon,
            tile_lat=tile_lat,
            size_km=size_km,
            observed_at=observed_at,
            payload=canonical,
            payload_hash_value=current_hash,
            change_type="cleared",
            image_bytes=image_bytes,
            diff=diff,
        )
        return {"transmitted": True, "packet": packet}

    if cached is None:
        _store[cache_key] = (canonical, current_hash[:16])
        packet = _build_packet(
            tile_id=tile_id,
            tile_lon=tile_lon,
            tile_lat=tile_lat,
            size_km=size_km,
            observed_at=observed_at,
            payload=canonical,
            payload_hash_value=current_hash,
            change_type="first_observation",
            image_bytes=image_bytes,
        )
        return {"transmitted": True, "packet": packet}

    previous_payload, previous_hash = cached
    if current_hash[:16] == previous_hash:
        return {"transmitted": False, "reason": "hash_match"}

    diff = field_level_diff(canonical, previous_payload)
    _store[cache_key] = (canonical, current_hash[:16])
    if not has_meaningful_change(diff):
        return {"transmitted": False, "reason": "no_meaningful_change"}

    packet = _build_packet(
        tile_id=tile_id,
        tile_lon=tile_lon,
        tile_lat=tile_lat,
        size_km=size_km,
        observed_at=observed_at,
        payload=canonical,
        payload_hash_value=current_hash,
        change_type="updated",
        image_bytes=image_bytes,
        diff=diff,
    )
    return {"transmitted": True, "packet": packet}


def append_packet(packet_path: Path, packet: dict[str, object]) -> None:
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(packet, sort_keys=True))
        fh.write("\n")

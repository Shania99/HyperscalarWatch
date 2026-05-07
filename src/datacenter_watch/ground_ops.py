"""Ground-side packet ingest, site matching, and alert persistence."""

from __future__ import annotations

import base64
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datacenter_watch.downlink import (
    detection_centroid_lon_lat,
    field_level_diff,
    has_meaningful_change,
)

MATCH_RADIUS_METERS = 500.0
CONSTRUCTION_STAGE_ORDER = {
    "undisturbed": 0,
    "land_clearing": 1,
    "earthworks": 2,
    "foundations": 3,
    "structural_shell": 4,
    "roof_complete": 5,
    "operational": 6,
    "expansion": 7,
}
ACTIVE_CONSTRUCTION_STAGES = {
    "land_clearing",
    "earthworks",
    "foundations",
    "structural_shell",
    "roof_complete",
    "expansion",
}
OBSERVATION_IMAGE_COLUMNS = ("rgb_path", "swir_path", "index_path", "mapbox_path")
_ASSET_DIRS: dict[int, Path] = {}


def init_ground_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ASSET_DIRS[id(conn)] = path.parent / f"{path.stem}_assets"
    _ASSET_DIRS[id(conn)].mkdir(parents=True, exist_ok=True)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS downlink_packets (
            packet_id    TEXT PRIMARY KEY,
            tile_id      TEXT NOT NULL,
            observed_at  TEXT NOT NULL,
            packet_json  TEXT NOT NULL,
            processed_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sites (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            lon                        REAL NOT NULL,
            lat                        REAL NOT NULL,
            first_seen_at              TEXT NOT NULL,
            last_seen_at               TEXT NOT NULL,
            active                     INTEGER NOT NULL DEFAULT 1,
            current_site_class         TEXT,
            current_construction_stage TEXT,
            current_payload_json       TEXT NOT NULL,
            current_tile_context_json  TEXT NOT NULL,
            current_hash               TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id            INTEGER NOT NULL,
            tile_id            TEXT NOT NULL,
            observed_at        TEXT NOT NULL,
            event_type         TEXT NOT NULL,
            lon                REAL NOT NULL,
            lat                REAL NOT NULL,
            bbox_json          TEXT,
            detection_json     TEXT,
            tile_context_json  TEXT NOT NULL,
            packet_hash        TEXT NOT NULL,
            packet_json        TEXT NOT NULL,
            rgb_path           TEXT,
            swir_path          TEXT,
            index_path         TEXT,
            mapbox_path        TEXT,
            created_at         TEXT NOT NULL,
            FOREIGN KEY(site_id) REFERENCES sites(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stage_history (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id            INTEGER NOT NULL,
            observation_id     INTEGER NOT NULL,
            construction_stage TEXT NOT NULL,
            observed_at        TEXT NOT NULL,
            FOREIGN KEY(site_id) REFERENCES sites(id),
            FOREIGN KEY(observation_id) REFERENCES observations(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id        INTEGER NOT NULL,
            observation_id INTEGER NOT NULL,
            severity       TEXT NOT NULL,
            alert_type     TEXT NOT NULL,
            summary        TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            FOREIGN KEY(site_id) REFERENCES sites(id),
            FOREIGN KEY(observation_id) REFERENCES observations(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_queue (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id       INTEGER NOT NULL,
            status         TEXT NOT NULL,
            payload_json   TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            FOREIGN KEY(alert_id) REFERENCES alerts(id)
        )
    """)
    _ensure_observation_columns(conn)
    conn.commit()
    return conn


def _asset_dir(conn: sqlite3.Connection) -> Path:
    return _ASSET_DIRS[id(conn)]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_observation_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "observations")
    for column_name in OBSERVATION_IMAGE_COLUMNS:
        if column_name in columns:
            continue
        conn.execute(f"ALTER TABLE observations ADD COLUMN {column_name} TEXT")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def _site_row_to_payload(row: sqlite3.Row) -> dict[str, object]:
    return {
        "detections": [json.loads(str(row["current_payload_json"]))],
        "tile_context": json.loads(str(row["current_tile_context_json"])),
    }


def _find_site(conn: sqlite3.Connection, lon: float, lat: float) -> sqlite3.Row | None:
    best_row: sqlite3.Row | None = None
    best_distance = float("inf")
    for row in conn.execute("SELECT * FROM sites"):
        distance = _haversine_meters(lon, lat, float(row["lon"]), float(row["lat"]))
        if distance <= MATCH_RADIUS_METERS and distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row


def _packet_without_images(packet: dict[str, object]) -> dict[str, object]:
    stripped = dict(packet)
    stripped.pop("images", None)
    return stripped


def _image_extension(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return "jpg"
    return "png"


def _write_packet_images(
    conn: sqlite3.Connection,
    packet: dict[str, object],
) -> dict[str, str | None]:
    image_paths: dict[str, str | None] = {name: None for name in ("rgb", "swir", "index", "mapbox")}
    images_value = packet.get("images")
    images = images_value if isinstance(images_value, dict) else {}
    if not images:
        return image_paths

    packet_dir = _asset_dir(conn) / str(packet["packet_id"])
    packet_dir.mkdir(parents=True, exist_ok=True)
    for name in image_paths:
        image_value = images.get(name)
        if not isinstance(image_value, dict):
            continue
        data_b64 = image_value.get("data_b64")
        mime_type = str(image_value.get("mime_type") or "image/png")
        if not isinstance(data_b64, str) or not data_b64:
            continue
        image_path = packet_dir / f"{name}.{_image_extension(mime_type)}"
        image_path.write_bytes(base64.b64decode(data_b64))
        image_paths[name] = str(image_path)
    return image_paths


def _insert_packet_log(conn: sqlite3.Connection, packet: dict[str, object]) -> bool:
    packet_value = json.dumps(_packet_without_images(packet), sort_keys=True)
    try:
        conn.execute(
            """
            INSERT INTO downlink_packets (packet_id, tile_id, observed_at, packet_json, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(packet["packet_id"]),
                str(packet["tile_id"]),
                str(packet["observed_at"]),
                packet_value,
                _utc_now(),
            ),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def _insert_site(
    conn: sqlite3.Connection,
    *,
    lon: float,
    lat: float,
    observed_at: str,
    detection: dict[str, object],
    tile_context: dict[str, object],
    packet_hash: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO sites (
            lon, lat, first_seen_at, last_seen_at, active,
            current_site_class, current_construction_stage,
            current_payload_json, current_tile_context_json, current_hash
        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
        """,
        (
            lon,
            lat,
            observed_at,
            observed_at,
            detection.get("site_class"),
            detection.get("construction_stage"),
            json.dumps(detection, sort_keys=True),
            json.dumps(tile_context, sort_keys=True),
            packet_hash,
        ),
    )
    return int(cursor.lastrowid)


def _update_site(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    lon: float,
    lat: float,
    observed_at: str,
    active: bool,
    detection: dict[str, object],
    tile_context: dict[str, object],
    packet_hash: str,
) -> None:
    conn.execute(
        """
        UPDATE sites
        SET lon = ?, lat = ?, last_seen_at = ?, active = ?,
            current_site_class = ?, current_construction_stage = ?,
            current_payload_json = ?, current_tile_context_json = ?, current_hash = ?
        WHERE id = ?
        """,
        (
            lon,
            lat,
            observed_at,
            int(active),
            detection.get("site_class"),
            detection.get("construction_stage"),
            json.dumps(detection, sort_keys=True),
            json.dumps(tile_context, sort_keys=True),
            packet_hash,
            site_id,
        ),
    )


def _insert_observation(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    tile_id: str,
    observed_at: str,
    event_type: str,
    lon: float,
    lat: float,
    bbox: object,
    detection: dict[str, object],
    tile_context: dict[str, object],
    packet: dict[str, object],
    image_paths: dict[str, str | None],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO observations (
            site_id, tile_id, observed_at, event_type, lon, lat,
            bbox_json, detection_json, tile_context_json, packet_hash,
            packet_json, rgb_path, swir_path, index_path, mapbox_path, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            site_id,
            tile_id,
            observed_at,
            event_type,
            lon,
            lat,
            json.dumps(bbox),
            json.dumps(detection, sort_keys=True),
            json.dumps(tile_context, sort_keys=True),
            str(packet["payload_hash"]),
            json.dumps(_packet_without_images(packet), sort_keys=True),
            image_paths.get("rgb"),
            image_paths.get("swir"),
            image_paths.get("index"),
            image_paths.get("mapbox"),
            _utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def _latest_stage_entry(conn: sqlite3.Connection, site_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM stage_history
        WHERE site_id = ?
        ORDER BY observed_at DESC, id DESC
        LIMIT 1
        """,
        (site_id,),
    ).fetchone()


def _insert_stage_history(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    observation_id: int,
    construction_stage: str,
    observed_at: str,
) -> None:
    latest = _latest_stage_entry(conn, site_id)
    if latest is not None and str(latest["construction_stage"]) == construction_stage:
        return
    conn.execute(
        """
        INSERT INTO stage_history (site_id, observation_id, construction_stage, observed_at)
        VALUES (?, ?, ?, ?)
        """,
        (site_id, observation_id, construction_stage, observed_at),
    )


def _insert_alert(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    observation_id: int,
    severity: str,
    alert_type: str,
    summary: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO alerts (site_id, observation_id, severity, alert_type, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (site_id, observation_id, severity, alert_type, summary, _utc_now()),
    )
    return int(cursor.lastrowid)


def _queue_enrichment(
    conn: sqlite3.Connection,
    *,
    alert_id: int,
    site_id: int,
    observation_id: int,
    summary: str,
) -> None:
    conn.execute(
        """
        INSERT INTO enrichment_queue (alert_id, status, payload_json, created_at)
        VALUES (?, 'pending', ?, ?)
        """,
        (
            alert_id,
            json.dumps(
                {
                    "site_id": site_id,
                    "observation_id": observation_id,
                    "summary": summary,
                },
                sort_keys=True,
            ),
            _utc_now(),
        ),
    )


def _create_alert(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    observation_id: int,
    severity: str,
    alert_type: str,
    summary: str,
) -> None:
    alert_id = _insert_alert(
        conn,
        site_id=site_id,
        observation_id=observation_id,
        severity=severity,
        alert_type=alert_type,
        summary=summary,
    )
    if severity == "high":
        _queue_enrichment(
            conn,
            alert_id=alert_id,
            site_id=site_id,
            observation_id=observation_id,
            summary=summary,
        )


def _impact_alerts(
    detection: dict[str, object],
    tile_context: dict[str, object],
) -> list[tuple[str, str, str]]:
    alerts: list[tuple[str, str, str]] = []
    stage = str(detection.get("construction_stage", ""))
    proximity = tile_context.get("residential_proximity")
    if stage in ACTIVE_CONSTRUCTION_STAGES and proximity in {"adjacent", "within_1km"}:
        severity = "high" if proximity == "adjacent" else "medium"
        alerts.append(
            (
                severity,
                "impact_assessment",
                f"Construction activity is visible with residential proximity={proximity}.",
            )
        )
    if bool(detection.get("water_feature_present")) and bool(tile_context.get("shared_water_body_nearby")):
        alerts.append(
            (
                "medium",
                "impact_assessment",
                "Facility activity overlaps with nearby shared water infrastructure.",
            )
        )
    return alerts


def _rapid_construction_alert(
    previous_stage: str | None,
    previous_observed_at: str | None,
    current_stage: str,
    current_observed_at: str,
) -> tuple[str, str, str] | None:
    if not previous_stage or not previous_observed_at:
        return None
    previous_rank = CONSTRUCTION_STAGE_ORDER.get(previous_stage)
    current_rank = CONSTRUCTION_STAGE_ORDER.get(current_stage)
    if previous_rank is None or current_rank is None or current_rank <= previous_rank:
        return None
    days = (_parse_timestamp(current_observed_at) - _parse_timestamp(previous_observed_at)).days
    if current_rank - previous_rank >= 2 and days <= 60:
        return (
            "high",
            "rapid_construction",
            f"Stage advanced from {previous_stage} to {current_stage} in {days} days.",
        )
    return None


def _process_detection(
    conn: sqlite3.Connection,
    *,
    packet: dict[str, object],
    detection: dict[str, object],
    tile_context: dict[str, object],
    image_paths: dict[str, str | None],
) -> dict[str, int]:
    lon, lat = detection_centroid_lon_lat(
        float(packet["tile_center_lon"]),
        float(packet["tile_center_lat"]),
        float(packet["size_km"]),
        detection.get("bbox"),
    )
    site_row = _find_site(conn, lon, lat)
    observed_at = str(packet["observed_at"])
    summary = {"new_sites": 0, "updated_sites": 0, "alerts": 0}
    site_payload = {"detections": [detection], "tile_context": tile_context}

    if site_row is None:
        site_id = _insert_site(
            conn,
            lon=lon,
            lat=lat,
            observed_at=observed_at,
            detection=detection,
            tile_context=tile_context,
            packet_hash=str(packet["payload_hash"]),
        )
        observation_id = _insert_observation(
            conn,
            site_id=site_id,
            tile_id=str(packet["tile_id"]),
            observed_at=observed_at,
            event_type="detected",
            lon=lon,
            lat=lat,
            bbox=detection.get("bbox"),
            detection=detection,
            tile_context=tile_context,
            packet=packet,
            image_paths=image_paths,
        )
        _insert_stage_history(
            conn,
            site_id=site_id,
            observation_id=observation_id,
            construction_stage=str(detection.get("construction_stage", "")),
            observed_at=observed_at,
        )
        _create_alert(
            conn,
            site_id=site_id,
            observation_id=observation_id,
            severity="medium",
            alert_type="new_site",
            summary=f"New site detected: {detection.get('site_class')}.",
        )
        summary["new_sites"] += 1
        summary["alerts"] += 1
        for severity, alert_type, alert_summary in _impact_alerts(detection, tile_context):
            _create_alert(
                conn,
                site_id=site_id,
                observation_id=observation_id,
                severity=severity,
                alert_type=alert_type,
                summary=alert_summary,
            )
            summary["alerts"] += 1
        return summary

    site_id = int(site_row["id"])
    previous_payload = _site_row_to_payload(site_row)
    diff = field_level_diff(site_payload, previous_payload)
    observation_id = _insert_observation(
        conn,
        site_id=site_id,
        tile_id=str(packet["tile_id"]),
        observed_at=observed_at,
        event_type="detected",
        lon=lon,
        lat=lat,
        bbox=detection.get("bbox"),
        detection=detection,
        tile_context=tile_context,
        packet=packet,
        image_paths=image_paths,
    )
    _update_site(
        conn,
        site_id=site_id,
        lon=lon,
        lat=lat,
        observed_at=observed_at,
        active=True,
        detection=detection,
        tile_context=tile_context,
        packet_hash=str(packet["payload_hash"]),
    )
    summary["updated_sites"] += 1

    latest_stage = _latest_stage_entry(conn, site_id)
    previous_stage = str(latest_stage["construction_stage"]) if latest_stage is not None else None
    previous_seen = str(latest_stage["observed_at"]) if latest_stage is not None else None
    _insert_stage_history(
        conn,
        site_id=site_id,
        observation_id=observation_id,
        construction_stage=str(detection.get("construction_stage", "")),
        observed_at=observed_at,
    )

    if has_meaningful_change(diff):
        _create_alert(
            conn,
            site_id=site_id,
            observation_id=observation_id,
            severity="medium",
            alert_type="state_change",
            summary=f"Site state changed: {json.dumps(diff, sort_keys=True)}",
        )
        summary["alerts"] += 1

    rapid = _rapid_construction_alert(
        previous_stage,
        previous_seen,
        str(detection.get("construction_stage", "")),
        observed_at,
    )
    if rapid is not None:
        severity, alert_type, alert_summary = rapid
        _create_alert(
            conn,
            site_id=site_id,
            observation_id=observation_id,
            severity=severity,
            alert_type=alert_type,
            summary=alert_summary,
        )
        summary["alerts"] += 1

    for severity, alert_type, alert_summary in _impact_alerts(detection, tile_context):
        _create_alert(
            conn,
            site_id=site_id,
            observation_id=observation_id,
            severity=severity,
            alert_type=alert_type,
            summary=alert_summary,
        )
        summary["alerts"] += 1
    return summary


def _process_removed_detection(
    conn: sqlite3.Connection,
    *,
    packet: dict[str, object],
    detection: dict[str, object],
    tile_context: dict[str, object],
    image_paths: dict[str, str | None],
) -> dict[str, int]:
    lon, lat = detection_centroid_lon_lat(
        float(packet["tile_center_lon"]),
        float(packet["tile_center_lat"]),
        float(packet["size_km"]),
        detection.get("bbox"),
    )
    site_row = _find_site(conn, lon, lat)
    if site_row is None:
        return {"new_sites": 0, "updated_sites": 0, "alerts": 0}

    site_id = int(site_row["id"])
    observed_at = str(packet["observed_at"])
    observation_id = _insert_observation(
        conn,
        site_id=site_id,
        tile_id=str(packet["tile_id"]),
        observed_at=observed_at,
        event_type="cleared",
        lon=lon,
        lat=lat,
        bbox=detection.get("bbox"),
        detection=detection,
        tile_context=tile_context,
        packet=packet,
        image_paths=image_paths,
    )
    _update_site(
        conn,
        site_id=site_id,
        lon=lon,
        lat=lat,
        observed_at=observed_at,
        active=False,
        detection=detection,
        tile_context=tile_context,
        packet_hash=str(packet["payload_hash"]),
    )
    _create_alert(
        conn,
        site_id=site_id,
        observation_id=observation_id,
        severity="medium",
        alert_type="site_cleared",
        summary="Previously detected site is no longer present in the tile payload.",
    )
    return {"new_sites": 0, "updated_sites": 1, "alerts": 1}


def ingest_packet(conn: sqlite3.Connection, packet: dict[str, object]) -> dict[str, int | bool]:
    if not _insert_packet_log(conn, packet):
        return {"processed": False, "new_sites": 0, "updated_sites": 0, "alerts": 0}

    tile_context_value = packet.get("tile_context")
    tile_context = tile_context_value if isinstance(tile_context_value, dict) else {}
    image_paths = _write_packet_images(conn, packet)
    summary = {"processed": True, "new_sites": 0, "updated_sites": 0, "alerts": 0}

    detections_value = packet.get("detections")
    detections = detections_value if isinstance(detections_value, list) else []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        result = _process_detection(
            conn,
            packet=packet,
            detection=detection,
            tile_context=tile_context,
            image_paths=image_paths,
        )
        for key in ("new_sites", "updated_sites", "alerts"):
            summary[key] += result[key]

    diff_value = packet.get("diff")
    diff = diff_value if isinstance(diff_value, dict) else {}
    removed_items = diff.get("detections")
    if isinstance(removed_items, list):
        for change in removed_items:
            if not isinstance(change, dict) or change.get("type") != "removed":
                continue
            previous = change.get("previous")
            if not isinstance(previous, dict):
                continue
            result = _process_removed_detection(
                conn,
                packet=packet,
                detection=previous,
                tile_context=tile_context,
                image_paths=image_paths,
            )
            for key in ("new_sites", "updated_sites", "alerts"):
                summary[key] += result[key]

    conn.commit()
    return summary

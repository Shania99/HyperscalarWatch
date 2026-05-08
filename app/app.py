"""DataCenterWatch ground dashboard.

Run from the project root:
    uv run streamlit run app/app.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from PIL import Image, ImageDraw
from datacenter_watch.live import get_current_state
from datacenter_watch.locations import LOCATIONS

DB_PATH = Path(__file__).parent.parent / "ground.db"

STAGE_COLORS = {
    "operational": "#00e87a",
    "active_construction": "#ff9f1c",
    "undisturbed": "#64748b",
}
SEVERITY_COLORS = {"high": "#ff5577", "medium": "#ffaa00", "low": "#00e87a"}
STAGE_RANK = {
    "undisturbed": 0,
    "active_construction": 1,
    "operational": 2,
}
DETECTION_FIELDS = [
    "site_class",
    "construction_stage",
    "roof_bright_membrane",
    "bare_soil_present",
    "reasoning",
]
TILE_CONTEXT_FIELDS = [
    "image_quality_limited",
]


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        html, body, [class*="css"]  { font-family: 'JetBrains Mono', monospace; }
        .stApp {
            background:
              radial-gradient(circle at 12% 18%, rgba(0, 232, 122, 0.14), transparent 25%),
              radial-gradient(circle at 88% 14%, rgba(85, 87, 255, 0.16), transparent 24%),
              linear-gradient(180deg, #08111d 0%, #0b1423 45%, #09111b 100%);
            color: #e8eef8;
        }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
        .hero {
            border: 1px solid rgba(82, 116, 168, 0.35);
            background: linear-gradient(135deg, rgba(12, 20, 36, 0.95), rgba(15, 27, 45, 0.88));
            border-radius: 18px;
            padding: 20px 24px 18px 24px;
            margin-bottom: 18px;
            box-shadow: 0 22px 48px rgba(0, 0, 0, 0.28);
        }
        .hero-kicker { color: #7fa7d8; font-size: 11px; letter-spacing: 0.3em; margin-bottom: 8px; }
        .hero-title { font-size: 30px; font-weight: 700; color: #f4f8ff; margin-bottom: 8px; }
        .hero-subtitle { color: #9ab0c8; font-size: 13px; line-height: 1.6; max-width: 920px; }
        .metric-card {
            border: 1px solid rgba(82, 116, 168, 0.28);
            background: rgba(12, 20, 36, 0.86);
            border-radius: 14px;
            padding: 14px 16px;
            min-height: 110px;
        }
        .metric-label { font-size: 10px; color: #7fa7d8; letter-spacing: 0.24em; margin-bottom: 10px; }
        .metric-value { font-size: 28px; color: #f2f7ff; font-weight: 700; margin-bottom: 6px; }
        .metric-note { font-size: 11px; color: #8ca0ba; line-height: 1.5; }
        .panel {
            border: 1px solid rgba(82, 116, 168, 0.28);
            background: rgba(12, 20, 36, 0.82);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .section-kicker { color: #7fa7d8; font-size: 10px; letter-spacing: 0.28em; margin-bottom: 8px; }
        .section-title { color: #f2f7ff; font-size: 18px; font-weight: 700; margin-bottom: 12px; }
        .alert-card {
            border-left: 4px solid #ffaa00;
            background: rgba(16, 24, 42, 0.96);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 10px;
            letter-spacing: 0.16em;
            font-weight: 700;
            margin-right: 8px;
        }
        .kv-card {
            border: 1px solid rgba(82, 116, 168, 0.22);
            background: rgba(12, 20, 36, 0.76);
            border-radius: 12px;
            padding: 12px;
            height: 100%;
        }
        .kv-label { color: #7fa7d8; font-size: 10px; letter-spacing: 0.16em; margin-bottom: 7px; }
        .kv-value { color: #f4f8ff; font-size: 14px; line-height: 1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _parse_json(value: object, fallback: Any) -> Any:
    if not value:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return fallback


def _site_label(tile_id: str, site_id: int) -> str:
    location = tile_id.split("/", 1)[0] if tile_id else f"site_{site_id}"
    return location


def _fmt_ts(value: str) -> str:
    if not value:
        return "n/a"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


def _overlay_image(path_value: str | None, bbox: object) -> Image.Image | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width, height = image.size
        draw = ImageDraw.Draw(image)
        left = int(x1 * width)
        top = int(y1 * height)
        right = int(x2 * width)
        bottom = int(y2 * height)
        draw.rectangle((left, top, right, bottom), outline="#ff5577", width=4)
    return image


def _metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _alert_card(row: dict[str, Any]) -> None:
    severity = str(row["severity"])
    color = SEVERITY_COLORS.get(severity, "#7fa7d8")
    active = "ACTIVE" if row["active"] else "INACTIVE"
    st.markdown(
        f"""
        <div class="alert-card" style="border-left-color:{color}">
          <div style="margin-bottom:7px">
            <span class="pill" style="background:{color}22;color:{color};border:1px solid {color}44;">{severity.upper()}</span>
            <span class="pill" style="background:#142038;color:#9ab0c8;border:1px solid rgba(82,116,168,0.28);">{str(row["alert_type"]).upper()}</span>
            <span class="pill" style="background:#10182a;color:#cbd6e5;border:1px solid rgba(82,116,168,0.20);">{active}</span>
          </div>
          <div style="color:#f2f7ff;font-size:13px;font-weight:600;margin-bottom:6px;">{row["site_name"]}</div>
          <div style="color:#d0dae8;font-size:12px;line-height:1.55;margin-bottom:8px;">{row["summary"]}</div>
          <div style="color:#7fa7d8;font-size:10px;letter-spacing:0.12em;">{_fmt_ts(str(row["created_at"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fetch_latest_watch_task(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT packet_id, tile_id, observed_at, processed_at, packet_json
        FROM downlink_packets
        WHERE tile_id NOT LIKE 'live/%'
        ORDER BY processed_at DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    packet = _parse_json(row["packet_json"], {})
    if not isinstance(packet, dict):
        return None
    try:
        lon = float(packet["tile_center_lon"])
        lat = float(packet["tile_center_lat"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "packet_id": str(row["packet_id"]),
        "tile_id": str(row["tile_id"]),
        "observed_at": str(row["observed_at"]),
        "processed_at": str(row["processed_at"]),
        "lon": lon,
        "lat": lat,
    }


def _globe_figure(
    sites: list[dict[str, Any]],
    selected_site_id: int | None,
    satellite_now: dict[str, Any] | None,
    live_satellite: dict[str, Any] | None,
) -> go.Figure:
    lons = [float(site["lon"]) for site in sites]
    lats = [float(site["lat"]) for site in sites]
    texts = [
        (
            f"{site['site_name']}<br>"
            f"class: {site['site_class']}<br>"
            f"stage: {site['construction_stage']}<br>"
            f"last seen: {_fmt_ts(str(site['observed_at']))}"
        )
        for site in sites
    ]
    colors = [
        "#ffffff" if site["site_id"] == selected_site_id else site["marker_color"]
        for site in sites
    ]
    sizes = [
        17 if site["site_id"] == selected_site_id else site["marker_size"]
        for site in sites
    ]
    line_colors = [
        site["marker_color"] if site["site_id"] == selected_site_id else "#08111d"
        for site in sites
    ]

    center_lon = float(sites[0]["lon"]) if sites else -96.0
    center_lat = float(sites[0]["lat"]) if sites else 37.0
    selected = next((site for site in sites if site["site_id"] == selected_site_id), None)
    if selected is not None:
        center_lon = float(selected["lon"])
        center_lat = float(selected["lat"])

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=[loc.lon for loc in LOCATIONS],
            lat=[loc.lat for loc in LOCATIONS],
            text=[loc.id for loc in LOCATIONS],
            hovertemplate="watchlist: %{text}<extra></extra>",
            mode="markers",
            marker={
                "size": 4,
                "color": "#6c86a4",
                "opacity": 0.28,
                "line": {"color": "#112133", "width": 0},
            },
            name="Watchlist",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            text=texts,
            hoverinfo="text",
            mode="markers",
            marker={
                "size": sizes,
                "color": colors,
                "line": {"color": line_colors, "width": 2},
                "opacity": 0.96,
            },
        )
    )
    if satellite_now is not None:
        fig.add_trace(
            go.Scattergeo(
                lon=[float(satellite_now["lon"])],
                lat=[float(satellite_now["lat"])],
                text=[
                    (
                        "latest watch task<br>"
                        f"tile: {satellite_now['tile_id']}<br>"
                        f"observed: {_fmt_ts(str(satellite_now['observed_at']))}<br>"
                        f"processed: {_fmt_ts(str(satellite_now['processed_at']))}"
                    )
                ],
                hoverinfo="text",
                mode="markers",
                marker={
                    "size": 20,
                    "color": "#ff9f43",
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "line": {"color": "#fff5cc", "width": 2},
                },
                name="Latest Watch Task",
            )
        )
    if live_satellite is not None:
        fig.add_trace(
            go.Scattergeo(
                lon=[float(live_satellite["lon"])],
                lat=[float(live_satellite["lat"])],
                text=[
                    (
                        "simsat live position<br>"
                        f"timestamp: {_fmt_ts(str(live_satellite.get('timestamp', '')))}<br>"
                        f"alt: {float(live_satellite.get('alt_km', 0.0)):.1f} km"
                    )
                ],
                hoverinfo="text",
                mode="markers",
                marker={
                    "size": 18,
                    "color": "#4cc9f0",
                    "symbol": "star",
                    "opacity": 1.0,
                    "line": {"color": "#d6f6ff", "width": 2},
                },
                name="SimSat Live",
            )
        )
    fig.update_geos(
        projection={"type": "orthographic", "rotation": {"lon": center_lon, "lat": center_lat}},
        showland=True,
        landcolor="#17314e",
        showocean=True,
        oceancolor="#081626",
        showlakes=True,
        lakecolor="#081626",
        showcountries=True,
        countrycolor="#35597f",
        coastlinecolor="#4d7aa4",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
    )
    return fig


def _stage_timeline_figure(history: list[dict[str, Any]]) -> go.Figure:
    xs = [row["observed_at"] for row in history]
    ys = [STAGE_RANK.get(str(row["construction_stage"]), 0) for row in history]
    labels = [str(row["construction_stage"]) for row in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": "#ff5577", "width": 3},
            marker={"size": 8, "color": "#ffaa00"},
            text=labels,
            hovertemplate="%{text}<br>%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 40, "r": 10, "t": 10, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={
            "tickvals": list(STAGE_RANK.values()),
            "ticktext": list(STAGE_RANK.keys()),
            "gridcolor": "rgba(82,116,168,0.18)",
        },
        xaxis={"gridcolor": "rgba(82,116,168,0.12)"},
        font={"color": "#d5deeb"},
    )
    return fig


def _fetch_cutoff_options(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT observed_at FROM observations ORDER BY observed_at"
    ).fetchall()
    return [str(row["observed_at"]) for row in rows]


def _fetch_sites(conn: sqlite3.Connection, cutoff: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            s.id AS site_id,
            s.first_seen_at,
            o.id AS observation_id,
            o.tile_id,
            o.observed_at,
            o.event_type,
            o.lon,
            o.lat,
            o.bbox_json,
            o.detection_json,
            o.tile_context_json,
            o.rgb_path,
            o.swir_path,
            o.index_path,
            o.mapbox_path
        FROM sites s
        JOIN observations o
          ON o.id = (
              SELECT id
              FROM observations
              WHERE site_id = s.id AND observed_at <= ?
              ORDER BY observed_at DESC, id DESC
              LIMIT 1
          )
        ORDER BY o.observed_at DESC, s.id DESC
        """,
        (cutoff,),
    ).fetchall()
    sites: list[dict[str, Any]] = []
    for row in rows:
        site = dict(row)
        detection = _parse_json(site.pop("detection_json"), {})
        tile_context = _parse_json(site.pop("tile_context_json"), {})
        bbox = _parse_json(site.pop("bbox_json"), None)
        site["detection"] = detection
        site["tile_context"] = tile_context
        site["bbox"] = bbox
        site["site_class"] = str(detection.get("site_class") or "unknown")
        site["construction_stage"] = str(detection.get("construction_stage") or "unknown")
        site["active"] = str(site["event_type"]) != "cleared"
        site["site_name"] = _site_label(str(site["tile_id"]), int(site["site_id"]))
        sites.append(site)
    return sites


def _fetch_live_satellite() -> dict[str, Any] | None:
    try:
        return get_current_state()
    except Exception:
        return None


def _fetch_alerts(conn: sqlite3.Connection, cutoff: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            a.id AS alert_id,
            a.site_id,
            a.observation_id,
            a.severity,
            a.alert_type,
            a.summary,
            a.created_at,
            o.tile_id,
            o.observed_at,
            o.lon,
            o.lat,
            o.event_type
        FROM alerts a
        JOIN observations o ON o.id = a.observation_id
        WHERE o.observed_at <= ?
        ORDER BY a.created_at DESC, a.id DESC
        """,
        (cutoff,),
    ).fetchall()
    alerts = [dict(row) for row in rows]
    return alerts


def _fetch_stage_history(conn: sqlite3.Connection, site_id: int, cutoff: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT construction_stage, observed_at
        FROM stage_history
        WHERE site_id = ? AND observed_at <= ?
        ORDER BY observed_at, id
        """,
        (site_id, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_observations(conn: sqlite3.Connection, site_id: int, cutoff: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id,
            tile_id,
            observed_at,
            event_type,
            lon,
            lat,
            bbox_json,
            detection_json,
            tile_context_json,
            rgb_path,
            swir_path,
            index_path,
            mapbox_path
        FROM observations
        WHERE site_id = ? AND observed_at <= ?
        ORDER BY observed_at DESC, id DESC
        """,
        (site_id, cutoff),
    ).fetchall()
    observations: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["bbox"] = _parse_json(item.pop("bbox_json"), None)
        item["detection"] = _parse_json(item.pop("detection_json"), {})
        item["tile_context"] = _parse_json(item.pop("tile_context_json"), {})
        observations.append(item)
    return observations


def _fetch_enrichment(conn: sqlite3.Connection, site_id: int, cutoff: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            q.status,
            q.payload_json,
            q.created_at,
            a.alert_type,
            a.severity,
            a.summary
        FROM enrichment_queue q
        JOIN alerts a ON a.id = q.alert_id
        JOIN observations o ON o.id = a.observation_id
        WHERE a.site_id = ? AND o.observed_at <= ?
        ORDER BY q.created_at DESC, q.id DESC
        """,
        (site_id, cutoff),
    ).fetchall()
    enrichments = [dict(row) for row in rows]
    for row in enrichments:
        row["payload"] = _parse_json(row.pop("payload_json"), {})
    return enrichments


def _decorate_sites(
    sites: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_alert_by_site: dict[int, dict[str, Any]] = {}
    for alert in alerts:
        site_id = int(alert["site_id"])
        if site_id not in latest_alert_by_site:
            latest_alert_by_site[site_id] = alert

    for site in sites:
        latest_alert = latest_alert_by_site.get(int(site["site_id"]))
        severity = str(latest_alert["severity"]) if latest_alert is not None else "low"
        site["latest_alert"] = latest_alert
        site["marker_color"] = SEVERITY_COLORS.get(severity, "#00e87a")
        site["marker_size"] = 14 if severity == "high" else 11 if severity == "medium" else 9
    return sites


def _render_kv_grid(items: list[tuple[str, Any]], columns: int = 3) -> None:
    groups = [items[index:index + columns] for index in range(0, len(items), columns)]
    for group in groups:
        cols = st.columns(len(group))
        for col, (label, value) in zip(cols, group, strict=False):
            with col:
                st.markdown(
                    f"""
                    <div class="kv-card">
                      <div class="kv-label">{label.upper()}</div>
                      <div class="kv-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.set_page_config(page_title="DataCenterWatch", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">GROUND OPS DASHBOARD</div>
          <div class="hero-title">DataCenterWatch</div>
          <div class="hero-subtitle">
            Track changed detections downlinked from the satellite runtime, inspect raw RGB / SWIR /
            index / Mapbox imagery, and triage site evolution with alert history, stage history,
            and enrichment status in one place.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DB_PATH.exists():
        st.error(f"Ground database not found: {DB_PATH}")
        st.caption("Run `uv run scripts/ground.py --downlink <packets.jsonl> --db ground.db` first.")
        return

    conn = _connect(DB_PATH)
    cutoff_options = _fetch_cutoff_options(conn)
    if not cutoff_options:
        st.info("No ground observations yet. Ingest packets into `ground.db` first.")
        return
    satellite_now = _fetch_latest_watch_task(conn)
    live_satellite = _fetch_live_satellite()

    with st.sidebar:
        st.header("Controls")
        cutoff = st.select_slider(
            "Observation timeline",
            options=cutoff_options,
            value=cutoff_options[-1],
            format_func=_fmt_ts,
        )
        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_interval = st.number_input(
            "Refresh every (seconds)", min_value=3, max_value=300, value=10, step=1
        )
        if st.button("Refresh now"):
            st.rerun()

    if auto_refresh:
        st_autorefresh(interval=int(refresh_interval) * 1000, key="auto_refresh")

    sites = _fetch_sites(conn, cutoff)
    alerts = _fetch_alerts(conn, cutoff)
    sites = _decorate_sites(sites, alerts)
    if not sites:
        st.info("No sites available at the selected timeline position.")
        return

    severity_options = sorted({str(alert["severity"]) for alert in alerts}, key=lambda v: {"high": 0, "medium": 1, "low": 2}.get(v, 9))
    alert_types = sorted({str(alert["alert_type"]) for alert in alerts})
    site_classes = sorted({str(site["site_class"]) for site in sites})

    with st.sidebar:
        severity_filter = st.multiselect("Severity", severity_options, default=severity_options)
        alert_type_filter = st.multiselect("Alert type", alert_types, default=alert_types)
        site_class_filter = st.multiselect("Site class", site_classes, default=site_classes)
        active_only = st.toggle("Active sites only", value=False)

    filtered_sites = [
        site
        for site in sites
        if site["site_class"] in site_class_filter and (site["active"] or not active_only)
    ]
    filtered_alerts = [
        {
            **alert,
            "site_name": next(
                (site["site_name"] for site in sites if int(site["site_id"]) == int(alert["site_id"])),
                f"site_{alert['site_id']}",
            ),
            "active": next(
                (site["active"] for site in sites if int(site["site_id"]) == int(alert["site_id"])),
                False,
            ),
        }
        for alert in alerts
        if alert["severity"] in severity_filter and alert["alert_type"] in alert_type_filter
    ]
    filtered_alerts = [alert for alert in filtered_alerts if any(int(site["site_id"]) == int(alert["site_id"]) for site in filtered_sites)]
    if not filtered_sites:
        st.warning("No sites remain after filtering.")
        return

    default_site = filtered_sites[0]
    alert_site_ids = [int(alert["site_id"]) for alert in filtered_alerts]
    if alert_site_ids:
        default_site = next((site for site in filtered_sites if int(site["site_id"]) == alert_site_ids[0]), default_site)

    site_index = {f"{site['site_name']} · #{site['site_id']}": site for site in filtered_sites}
    with st.sidebar:
        selected_label = st.selectbox("Focus site", list(site_index), index=list(site_index).index(f"{default_site['site_name']} · #{default_site['site_id']}"))
    selected_site = site_index[selected_label]

    active_count = sum(1 for site in filtered_sites if site["active"])
    high_alert_count = sum(1 for alert in filtered_alerts if str(alert["severity"]) == "high")
    rapid_count = sum(1 for alert in filtered_alerts if str(alert["alert_type"]) == "rapid_construction")
    pending_enrichment = conn.execute(
        """
        SELECT COUNT(*)
        FROM enrichment_queue q
        JOIN alerts a ON a.id = q.alert_id
        JOIN observations o ON o.id = a.observation_id
        WHERE q.status = 'pending' AND o.observed_at <= ?
        """,
        (cutoff,),
    ).fetchone()[0]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card("ACTIVE SITES", str(active_count), "Sites whose latest observation before the cutoff is still active.")
    with metric_cols[1]:
        _metric_card("HIGH ALERTS", str(high_alert_count), "Newest operational changes that likely need immediate human review.")
    with metric_cols[2]:
        _metric_card("RAPID BUILDS", str(rapid_count), "Stage jumps detected inside the rapid-construction rule window.")
    with metric_cols[3]:
        _metric_card("ENRICHMENT QUEUE", str(pending_enrichment), "Queued high-severity alerts waiting for external narrative enrichment.")

    if satellite_now is not None or live_satellite is not None:
        parts: list[str] = []
        if live_satellite is not None:
            parts.append(
                f"SimSat live position: <b>{float(live_satellite['lat']):.5f}, {float(live_satellite['lon']):.5f}</b>"
                f" at {_fmt_ts(str(live_satellite.get('timestamp', '')))}."
            )
        if satellite_now is not None:
            parts.append(
                f"Latest processed watch task: <b>{satellite_now['tile_id']}</b>"
                f" at {_fmt_ts(str(satellite_now['processed_at']))}."
            )
        st.markdown(
            f"""
            <div class="panel">
              <div class="section-kicker">LIVE TRACK</div>
              <div class="section-title">SimSat Live + Watch Task</div>
              <div style="color:#d0dae8;font-size:12px;line-height:1.7;">
                {' '.join(parts)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="section-kicker">GLOBAL VIEW</div><div class="section-title">Tracked Sites</div></div>', unsafe_allow_html=True)
        st.plotly_chart(
            _globe_figure(filtered_sites, int(selected_site["site_id"]), satellite_now, live_satellite),
            use_container_width=True,
        )
    with right:
        st.markdown('<div class="panel"><div class="section-kicker">ALERT INBOX</div><div class="section-title">Latest Changes</div></div>', unsafe_allow_html=True)
        for alert in filtered_alerts[:8]:
            _alert_card(alert)
        if not filtered_alerts:
            st.caption("No alerts match the current filters.")

    st.markdown(
        f"""
        <div class="panel">
          <div class="section-kicker">SITE DETAIL</div>
          <div class="section-title">{selected_site['site_name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    site_stage = str(selected_site["construction_stage"])
    site_stage_color = STAGE_COLORS.get(site_stage, "#7fa7d8")
    stage_html = (
        f'<span class="pill" style="background:{site_stage_color}22;color:{site_stage_color};border:1px solid {site_stage_color}44;">'
        f'{site_stage.upper()}</span>'
    )
    active_html = (
        '<span class="pill" style="background:#00e87a22;color:#00e87a;border:1px solid #00e87a44;">ACTIVE</span>'
        if selected_site["active"]
        else '<span class="pill" style="background:#ff557722;color:#ff5577;border:1px solid #ff557744;">INACTIVE</span>'
    )
    st.markdown(stage_html + active_html, unsafe_allow_html=True)

    current_detection = selected_site["detection"]
    current_context = selected_site["tile_context"]
    current_bbox = selected_site["bbox"]
    observations = _fetch_observations(conn, int(selected_site["site_id"]), cutoff)
    history = _fetch_stage_history(conn, int(selected_site["site_id"]), cutoff)
    enrichment = _fetch_enrichment(conn, int(selected_site["site_id"]), cutoff)
    selected_alerts = [alert for alert in filtered_alerts if int(alert["site_id"]) == int(selected_site["site_id"])]

    summary_cols = st.columns(4)
    summary_items = [
        ("site class", selected_site["site_class"]),
        ("latest event", selected_site["event_type"]),
        ("last seen", _fmt_ts(str(selected_site["observed_at"]))),
        ("coordinates", f"{float(selected_site['lat']):.5f}, {float(selected_site['lon']):.5f}"),
    ]
    for col, (label, value) in zip(summary_cols, summary_items, strict=False):
        with col:
            _render_kv_grid([(label, value)], columns=1)

    tab_images, tab_timeline, tab_detection, tab_context, tab_enrichment = st.tabs(
        ["Imagery", "Timeline", "Detection", "Tile Context", "Enrichment"]
    )

    with tab_images:
        image_specs = [
            ("RGB", selected_site.get("rgb_path")),
            ("Mapbox", selected_site.get("mapbox_path")),
            ("SWIR", selected_site.get("swir_path")),
            ("Index", selected_site.get("index_path")),
        ]
        grid = st.columns(2)
        for idx, (label, path_value) in enumerate(image_specs):
            with grid[idx % 2]:
                st.markdown(f"**{label}**")
                rendered = _overlay_image(path_value, current_bbox)
                if rendered is None:
                    st.caption("No image available for this observation.")
                else:
                    st.image(rendered, use_container_width=True)

    with tab_timeline:
        top, bottom = st.columns([1.2, 1])
        with top:
            if history:
                st.plotly_chart(_stage_timeline_figure(history), use_container_width=True)
            else:
                st.caption("No stage history available.")
        with bottom:
            st.markdown("**Observation log**")
            for observation in observations[:10]:
                color = "#ff5577" if observation["event_type"] == "cleared" else "#00e87a"
                st.markdown(
                    f"""
                    <div class="alert-card" style="border-left-color:{color}">
                      <div style="color:#f2f7ff;font-size:12px;font-weight:600;">{observation['event_type'].upper()}</div>
                      <div style="color:#9ab0c8;font-size:11px;margin-top:5px;">{_fmt_ts(str(observation['observed_at']))}</div>
                      <div style="color:#d0dae8;font-size:11px;margin-top:7px;">{observation['tile_id']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if selected_alerts:
            st.markdown("**Alert history**")
            for alert in selected_alerts[:10]:
                _alert_card(alert)

    with tab_detection:
        detection_items = [(field.replace("_", " "), current_detection.get(field, "n/a")) for field in DETECTION_FIELDS]
        _render_kv_grid(detection_items, columns=3)

    with tab_context:
        context_items = [(field.replace("_", " "), current_context.get(field, "n/a")) for field in TILE_CONTEXT_FIELDS]
        _render_kv_grid(context_items, columns=3)

    with tab_enrichment:
        if enrichment:
            for row in enrichment:
                severity_color = SEVERITY_COLORS.get(str(row["severity"]), "#7fa7d8")
                st.markdown(
                    f"""
                    <div class="alert-card" style="border-left-color:{severity_color}">
                      <div style="margin-bottom:7px">
                        <span class="pill" style="background:{severity_color}22;color:{severity_color};border:1px solid {severity_color}44;">{str(row['severity']).upper()}</span>
                        <span class="pill" style="background:#142038;color:#9ab0c8;border:1px solid rgba(82,116,168,0.28);">{str(row['status']).upper()}</span>
                      </div>
                      <div style="color:#f2f7ff;font-size:13px;font-weight:600;margin-bottom:6px;">{row['alert_type']}</div>
                      <div style="color:#d0dae8;font-size:12px;line-height:1.55;margin-bottom:8px;">{row['summary']}</div>
                      <div style="color:#7fa7d8;font-size:10px;letter-spacing:0.12em;">{_fmt_ts(str(row['created_at']))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No enrichment queue items for this site yet.")



if __name__ == "__main__":
    main()

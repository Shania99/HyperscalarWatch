"""Fetch the current simulated satellite state and imagery from SimSat."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import requests

from datacenter_watch.simsat import (
    INDEX_BANDS,
    RGB_BANDS,
    SIMSAT_BASE_URL,
    SimSatNoImageError,
    SWIR_BANDS,
    _check_cloud_cover,
    build_index_composite,
)


def get_current_state(base_url: str = SIMSAT_BASE_URL) -> dict[str, Any]:
    """Return current simulated satellite telemetry from SimSat."""
    response = requests.get(f"{base_url}/data/current/position", timeout=30)
    response.raise_for_status()
    data = response.json()
    lon_lat_alt = data.get("lon-lat-alt")
    if not isinstance(lon_lat_alt, list) or len(lon_lat_alt) != 3:
        raise ValueError("SimSat current position payload is malformed")
    return {
        "lon": float(lon_lat_alt[0]),
        "lat": float(lon_lat_alt[1]),
        "alt_km": float(lon_lat_alt[2]),
        "timestamp": str(data.get("timestamp", "")),
    }


def current_state_is_ready(state: Mapping[str, object]) -> bool:
    timestamp = str(state.get("timestamp", ""))
    lon = float(state.get("lon", 0.0))
    lat = float(state.get("lat", 0.0))
    alt_km = float(state.get("alt_km", 0.0))
    return not (
        abs(lon) < 1e-9
        and abs(lat) < 1e-9
        and abs(alt_km) < 1e-9
        and timestamp in {"", "1970-01-01T00:00:00Z"}
    )


def get_current_position(base_url: str = SIMSAT_BASE_URL) -> tuple[float, float]:
    state = get_current_state(base_url)
    return float(state["lon"]), float(state["lat"])


def _load_json_header(response: requests.Response, header_name: str) -> dict[str, object]:
    raw = response.headers.get(header_name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _fetch_current_sentinel(
    bands: Sequence[str],
    size_km: float,
    *,
    return_type: str,
    base_url: str = SIMSAT_BASE_URL,
) -> requests.Response:
    params: list[tuple[str, object]] = [
        ("size_km", size_km),
        ("return_type", return_type),
    ] + [("spectral_bands", band) for band in bands]
    response = requests.get(
        f"{base_url}/data/current/image/sentinel",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response


def _fetch_current_png_with_metadata(
    bands: Sequence[str],
    size_km: float,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    response = _fetch_current_sentinel(bands, size_km, return_type="png", base_url=base_url)
    metadata = _load_json_header(response, "sentinel_metadata")
    if not response.content and metadata.get("image_available") is False:
        raise SimSatNoImageError(metadata)
    _check_cloud_cover(metadata)
    return response.content, metadata


def _fetch_current_array_with_metadata(
    bands: Sequence[str],
    size_km: float,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    response = _fetch_current_sentinel(bands, size_km, return_type="array", base_url=base_url)
    payload = response.json()
    image_payload = payload.get("image")
    sentinel_metadata = payload.get("sentinel_metadata")
    if image_payload is None:
        if isinstance(sentinel_metadata, Mapping) and sentinel_metadata.get("image_available") is False:
            raise SimSatNoImageError(sentinel_metadata)
        raise ValueError("SimSat current array response missing image payload")
    if not isinstance(image_payload, Mapping):
        raise ValueError("SimSat current array response missing image payload")
    meta = image_payload.get("metadata")
    image_b64 = image_payload.get("image")
    if not isinstance(meta, Mapping) or not isinstance(image_b64, str):
        raise ValueError("SimSat current array response is malformed")

    shape = tuple(meta.get("shape", ()))
    dtype = str(meta.get("dtype", ""))
    meta_bands = meta.get("bands")
    if len(shape) != 3 or not isinstance(meta_bands, list):
        raise ValueError("SimSat current array metadata is malformed")

    array = np.frombuffer(base64.b64decode(image_b64), dtype=np.dtype(dtype)).reshape(shape)
    bands_dict = {
        str(band): array[idx].astype(np.float32, copy=False)
        for idx, band in enumerate(meta_bands)
    }
    if not isinstance(sentinel_metadata, Mapping):
        sentinel_metadata = _load_json_header(response, "sentinel_metadata")
    _check_cloud_cover(sentinel_metadata)
    return bands_dict, dict(sentinel_metadata)


def fetch_live_rgb_with_metadata(
    size_km: float = 5.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    return _fetch_current_png_with_metadata(RGB_BANDS, size_km, base_url)


def fetch_live_swir_with_metadata(
    size_km: float = 5.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    return _fetch_current_png_with_metadata(SWIR_BANDS, size_km, base_url)


def fetch_live_index_with_metadata(
    size_km: float = 5.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    arrays, metadata = _fetch_current_array_with_metadata(INDEX_BANDS, size_km, base_url)
    return build_index_composite(arrays), metadata


def fetch_live_mapbox_with_metadata(
    target_lon: float | None = None,
    target_lat: float | None = None,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    params: dict[str, object] = {}
    if target_lon is not None:
        params["lon"] = target_lon
    if target_lat is not None:
        params["lat"] = target_lat
    response = requests.get(
        f"{base_url}/data/current/image/mapbox",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.content, _load_json_header(response, "mapbox_metadata")


def fetch_live_bundle(
    size_km: float = 5.0,
    base_url: str = SIMSAT_BASE_URL,
) -> dict[str, Any]:
    state = get_current_state(base_url)
    if not current_state_is_ready(state):
        raise ValueError(
            "SimSat current simulation is not ready: got lon=0, lat=0, alt=0 and epoch timestamp. "
            "Start or advance the SimSat simulation first."
        )
    rgb_bytes, rgb_meta = fetch_live_rgb_with_metadata(size_km, base_url)
    swir_bytes, _swir_meta = fetch_live_swir_with_metadata(size_km, base_url)
    index_bytes, _index_meta = fetch_live_index_with_metadata(size_km, base_url)
    mapbox_bytes = None
    mapbox_meta: dict[str, object] = {}
    try:
        mapbox_bytes, mapbox_meta = fetch_live_mapbox_with_metadata(
            float(state["lon"]),
            float(state["lat"]),
            base_url,
        )
        if not mapbox_bytes:
            mapbox_bytes = None
    except Exception:
        mapbox_bytes = None
        mapbox_meta = {}
    return {
        "state": state,
        "rgb_bytes": rgb_bytes,
        "rgb_metadata": rgb_meta,
        "swir_bytes": swir_bytes,
        "index_bytes": index_bytes,
        "mapbox_bytes": mapbox_bytes,
        "mapbox_metadata": mapbox_meta,
        "size_km": size_km,
    }

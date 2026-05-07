import base64
import io
import json
from collections.abc import Mapping, Sequence

import numpy as np
import requests
from PIL import Image

SIMSAT_BASE_URL = "http://localhost:9005"

RGB_BANDS = ("red", "green", "blue")
SWIR_BANDS = ("swir22", "nir08", "red")
INDEX_BANDS = ("red", "green", "nir08", "swir16")


class SimSatNoImageError(ValueError):
    """Raised when SimSat reports no Sentinel image for the requested query."""

    def __init__(self, metadata: Mapping[str, object] | None = None):
        self.metadata = dict(metadata or {})
        super().__init__("SimSat reported no Sentinel image for the requested query")


def _sentinel_params(
    lon: float,
    lat: float,
    timestamp: str,
    bands: Sequence[str],
    size_km: float,
    return_type: str,
) -> list[tuple[str, object]]:
    return [
        ("lon", lon),
        ("lat", lat),
        ("timestamp", timestamp),
        ("size_km", size_km),
        ("return_type", return_type),
    ] + [("spectral_bands", band) for band in bands]


def fetch_image(
    lon: float,
    lat: float,
    timestamp: str,
    bands: Sequence[str],
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Return raw PNG bytes for the requested Sentinel-2 composite."""
    response = requests.get(
        f"{base_url}/data/image/sentinel",
        params=_sentinel_params(lon, lat, timestamp, bands, size_km, "png"),
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def fetch_image_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    bands: Sequence[str],
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    response = requests.get(
        f"{base_url}/data/image/sentinel",
        params=_sentinel_params(lon, lat, timestamp, bands, size_km, "png"),
        timeout=60,
    )
    response.raise_for_status()
    return response.content, _load_json_header(response, "sentinel_metadata")


def fetch_image_array(
    lon: float,
    lat: float,
    timestamp: str,
    bands: Sequence[str],
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> dict[str, np.ndarray]:
    """Return raw Sentinel-2 bands as float arrays keyed by band name."""
    response = requests.get(
        f"{base_url}/data/image/sentinel",
        params=_sentinel_params(lon, lat, timestamp, bands, size_km, "array"),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    image_payload = payload.get("image")
    sentinel_metadata = payload.get("sentinel_metadata")
    if image_payload is None:
        if isinstance(sentinel_metadata, Mapping) and sentinel_metadata.get("image_available") is False:
            raise SimSatNoImageError(sentinel_metadata)
        raise ValueError("SimSat array response missing image payload")
    if not isinstance(image_payload, Mapping):
        raise ValueError("SimSat array response missing image payload")

    meta = image_payload.get("metadata")
    image_b64 = image_payload.get("image")
    if not isinstance(meta, Mapping) or not isinstance(image_b64, str):
        raise ValueError("SimSat array response is malformed")

    shape = tuple(meta.get("shape", ()))
    dtype = str(meta.get("dtype", ""))
    meta_bands = meta.get("bands")
    if len(shape) != 3 or not isinstance(meta_bands, list):
        raise ValueError("SimSat array metadata is malformed")

    array = np.frombuffer(base64.b64decode(image_b64), dtype=np.dtype(dtype)).reshape(shape)
    return {
        str(band): array[idx].astype(np.float32, copy=False)
        for idx, band in enumerate(meta_bands)
    }


def fetch_image_array_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    bands: Sequence[str],
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    response = requests.get(
        f"{base_url}/data/image/sentinel",
        params=_sentinel_params(lon, lat, timestamp, bands, size_km, "array"),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    image_payload = payload.get("image")
    sentinel_metadata = payload.get("sentinel_metadata")
    if image_payload is None:
        if isinstance(sentinel_metadata, Mapping) and sentinel_metadata.get("image_available") is False:
            raise SimSatNoImageError(sentinel_metadata)
        raise ValueError("SimSat array response missing image payload")
    if not isinstance(image_payload, Mapping):
        raise ValueError("SimSat array response missing image payload")

    meta = image_payload.get("metadata")
    image_b64 = image_payload.get("image")
    if not isinstance(meta, Mapping) or not isinstance(image_b64, str):
        raise ValueError("SimSat array response is malformed")

    shape = tuple(meta.get("shape", ()))
    dtype = str(meta.get("dtype", ""))
    meta_bands = meta.get("bands")
    if len(shape) != 3 or not isinstance(meta_bands, list):
        raise ValueError("SimSat array metadata is malformed")

    array = np.frombuffer(base64.b64decode(image_b64), dtype=np.dtype(dtype)).reshape(shape)
    bands_dict = {
        str(band): array[idx].astype(np.float32, copy=False)
        for idx, band in enumerate(meta_bands)
    }
    if not isinstance(sentinel_metadata, Mapping):
        sentinel_metadata = {}
    return bands_dict, dict(sentinel_metadata)


def fetch_rgb(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch a natural-color RGB composite (B4-B3-B2)."""
    return fetch_image(lon, lat, timestamp, RGB_BANDS, size_km, base_url)


def fetch_rgb_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    return fetch_image_with_metadata(lon, lat, timestamp, RGB_BANDS, size_km, base_url)


def fetch_swir(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch a SWIR composite (B12-B8-B4) for built surfaces and moisture contrast."""
    return fetch_image(lon, lat, timestamp, SWIR_BANDS, size_km, base_url)


def fetch_swir_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    return fetch_image_with_metadata(lon, lat, timestamp, SWIR_BANDS, size_km, base_url)


def fetch_index(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch the source bands and build the NDVI-MNDWI-NDBI composite."""
    arrays = fetch_image_array(lon, lat, timestamp, INDEX_BANDS, size_km, base_url)
    return build_index_composite(arrays)


def fetch_index_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    arrays, metadata = fetch_image_array_with_metadata(lon, lat, timestamp, INDEX_BANDS, size_km, base_url)
    return build_index_composite(arrays), metadata


def fetch_mapbox(
    target_lon: float,
    target_lat: float,
    satellite_lon: float | None = None,
    satellite_lat: float | None = None,
    satellite_alt: float = 800.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch a deterministic high-resolution Mapbox view for a target location."""
    if satellite_lon is None:
        satellite_lon = target_lon
    if satellite_lat is None:
        satellite_lat = target_lat
    response = requests.get(
        f"{base_url}/data/image/mapbox",
        params={
            "lon_target": target_lon,
            "lat_target": target_lat,
            "lon_satellite": satellite_lon,
            "lat_satellite": satellite_lat,
            "alt_satellite": satellite_alt,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def fetch_mapbox_with_metadata(
    target_lon: float,
    target_lat: float,
    satellite_lon: float | None = None,
    satellite_lat: float | None = None,
    satellite_alt: float = 800.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[bytes, dict[str, object]]:
    if satellite_lon is None:
        satellite_lon = target_lon
    if satellite_lat is None:
        satellite_lat = target_lat
    response = requests.get(
        f"{base_url}/data/image/mapbox",
        params={
            "lon_target": target_lon,
            "lat_target": target_lat,
            "lon_satellite": satellite_lon,
            "lat_satellite": satellite_lat,
            "alt_satellite": satellite_alt,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.content, _load_json_header(response, "mapbox_metadata")


def build_index_composite(bands: Mapping[str, np.ndarray]) -> bytes:
    """Build an RGB PNG where R=NDVI, G=MNDWI, B=NDBI."""
    required = {"red", "green", "nir08", "swir16"}
    missing = sorted(required - set(bands))
    if missing:
        raise ValueError(f"Missing bands for index composite: {', '.join(missing)}")

    ndvi = _normalized_difference(bands["nir08"], bands["red"])
    mndwi = _normalized_difference(bands["green"], bands["swir16"])
    ndbi = _normalized_difference(bands["swir16"], bands["nir08"])

    rgb = np.stack([
        _scale_index_to_uint8(ndvi),
        _scale_index_to_uint8(mndwi),
        _scale_index_to_uint8(ndbi),
    ], axis=-1)

    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    result = np.zeros_like(denom, dtype=np.float32)
    np.divide(a - b, denom, out=result, where=np.abs(denom) > 1e-6)
    return np.clip(result, -1.0, 1.0)


def _scale_index_to_uint8(values: np.ndarray) -> np.ndarray:
    return np.round((values + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _load_json_header(response: requests.Response, header_name: str) -> dict[str, object]:
    raw = response.headers.get(header_name, "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

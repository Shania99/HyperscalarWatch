"""Run satellite-side inference with RAM-cache suppression and JSONL downlink output."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datacenter_watch.evaluator import (
    PredictFn,
    anthropic_backend,
    gemini_backend,
    llama_backend,
    start_llama_server,
    stop_server,
    transformers_backend,
    wait_for_server,
)
from datacenter_watch.live import current_state_is_ready, fetch_live_bundle, get_current_state
from datacenter_watch.locations import LOCATIONS_BY_ID
from datacenter_watch.satellite_ops import append_packet, clear_cache, observe_tile
from datacenter_watch.simsat import (
    SimSatNoImageError,
    fetch_index_with_metadata,
    fetch_mapbox_with_metadata,
    fetch_rgb_with_metadata,
    fetch_swir_with_metadata,
)


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _choose_local_port(port: int) -> int:
    if _port_is_available(port):
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _center_from_footprint(footprint: object) -> tuple[float, float]:
    if not isinstance(footprint, list) or len(footprint) != 4:
        raise ValueError("imagery_metadata footprint is missing")
    lon_min, lat_min, lon_max, lat_max = (float(value) for value in footprint)
    return ((lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0)


def _sample_record(sample_dir: Path) -> dict[str, object]:
    metadata = json.loads((sample_dir / "imagery_metadata.json").read_text(encoding="utf-8"))
    rgb_meta = metadata.get("rgb")
    if not isinstance(rgb_meta, dict):
        raise ValueError(f"missing rgb metadata in {sample_dir}")
    lon, lat = _center_from_footprint(rgb_meta.get("footprint"))
    size_km = float(rgb_meta.get("size_km", 5.0))
    location_id = sample_dir.parent.name
    tile_name = sample_dir.name
    spatial_id = tile_name.split("_t", 1)[0]
    observed_at = str(metadata.get("requested_timestamp") or rgb_meta.get("datetime") or tile_name)
    return {
        "tile_id": f"{location_id}/{spatial_id}",
        "tile_lon": lon,
        "tile_lat": lat,
        "size_km": size_km,
        "observed_at": observed_at,
        "rgb_bytes": (sample_dir / "rgb.png").read_bytes(),
        "swir_bytes": (sample_dir / "swir.png").read_bytes(),
        "index_bytes": (sample_dir / "index.png").read_bytes() if (sample_dir / "index.png").exists() else None,
        "mapbox_bytes": (sample_dir / "mapbox.png").read_bytes() if (sample_dir / "mapbox.png").exists() else None,
        "label": f"{location_id}/{tile_name}",
    }


def _iter_sample_records(dataset_dir: Path) -> list[dict[str, object]]:
    sample_dirs = sorted(path.parent for path in dataset_dir.rglob("imagery_metadata.json"))
    records = [_sample_record(sample_dir) for sample_dir in sample_dirs]
    records.sort(key=lambda record: (str(record["tile_id"]), str(record["observed_at"])))
    return records


def _fetch_live_record(location_id: str, timestamp: str, size_km: float) -> dict[str, object]:
    location = LOCATIONS_BY_ID[location_id]
    rgb_bytes, rgb_meta = fetch_rgb_with_metadata(location.lon, location.lat, timestamp, size_km)
    swir_bytes, _swir_meta = fetch_swir_with_metadata(location.lon, location.lat, timestamp, size_km)
    index_bytes, _index_meta = fetch_index_with_metadata(location.lon, location.lat, timestamp, size_km)
    mapbox_bytes = None
    try:
        mapbox_bytes, _mapbox_meta = fetch_mapbox_with_metadata(location.lon, location.lat)
        if not mapbox_bytes:
            mapbox_bytes = None
    except Exception:
        mapbox_bytes = None
    lon, lat = _center_from_footprint(rgb_meta.get("footprint"))
    return {
        "tile_id": location_id,
        "tile_lon": lon,
        "tile_lat": lat,
        "size_km": size_km,
        "observed_at": timestamp,
        "rgb_bytes": rgb_bytes,
        "swir_bytes": swir_bytes,
        "index_bytes": index_bytes,
        "mapbox_bytes": mapbox_bytes,
        "label": location_id,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _live_tile_id(lon: float, lat: float, size_km: float) -> str:
    lat_step = size_km / 111.0
    safe_cos = max(0.2, abs(math.cos(math.radians(lat))))
    lon_step = size_km / (111.0 * safe_cos)
    lat_bucket = round(lat / lat_step)
    lon_bucket = round(lon / lon_step)
    return f"live/{lat_bucket}/{lon_bucket}"


def _watch_record(location_id: str, timestamp: str, size_km: float) -> dict[str, object]:
    location = LOCATIONS_BY_ID[location_id]
    rgb_bytes, rgb_meta = fetch_rgb_with_metadata(location.lon, location.lat, timestamp, size_km)
    swir_bytes, _swir_meta = fetch_swir_with_metadata(location.lon, location.lat, timestamp, size_km)
    index_bytes, _index_meta = fetch_index_with_metadata(location.lon, location.lat, timestamp, size_km)
    mapbox_bytes = None
    try:
        mapbox_bytes, _mapbox_meta = fetch_mapbox_with_metadata(location.lon, location.lat)
        if not mapbox_bytes:
            mapbox_bytes = None
    except Exception:
        mapbox_bytes = None
    lon, lat = _center_from_footprint(rgb_meta.get("footprint"))
    return {
        "tile_id": f"{location_id}/watch",
        "tile_lon": lon,
        "tile_lat": lat,
        "size_km": size_km,
        "observed_at": timestamp,
        "rgb_bytes": rgb_bytes,
        "swir_bytes": swir_bytes,
        "index_bytes": index_bytes,
        "mapbox_bytes": mapbox_bytes,
        "label": f"{location_id}/watch",
    }


def _current_record(size_km: float) -> dict[str, object]:
    bundle = fetch_live_bundle(size_km=size_km)
    state = bundle["state"]
    lon = float(state["lon"])
    lat = float(state["lat"])
    timestamp = str(state.get("timestamp") or _utc_now())
    return {
        "tile_id": _live_tile_id(lon, lat, size_km),
        "tile_lon": lon,
        "tile_lat": lat,
        "size_km": size_km,
        "observed_at": timestamp,
        "rgb_bytes": bundle["rgb_bytes"],
        "swir_bytes": bundle["swir_bytes"],
        "index_bytes": bundle["index_bytes"],
        "mapbox_bytes": bundle["mapbox_bytes"],
        "label": f"current/{lon:.3f},{lat:.3f}",
    }


def _process_record(
    record: dict[str, object],
    *,
    predict: PredictFn,
    downlink_path: Path,
) -> tuple[bool, str]:
    label = str(record["label"])
    print(f"[{label}] inferring ...")
    payload = predict(
        record["rgb_bytes"],  # type: ignore[arg-type]
        record["swir_bytes"],  # type: ignore[arg-type]
        index_bytes=record["index_bytes"],  # type: ignore[arg-type]
        mapbox_bytes=record["mapbox_bytes"],  # type: ignore[arg-type]
    )
    result = observe_tile(
        tile_id=str(record["tile_id"]),
        tile_lon=float(record["tile_lon"]),
        tile_lat=float(record["tile_lat"]),
        size_km=float(record["size_km"]),
        observed_at=str(record["observed_at"]),
        payload=payload,
        image_bytes={
            "rgb": record["rgb_bytes"],  # type: ignore[dict-item]
            "swir": record["swir_bytes"],  # type: ignore[dict-item]
            "index": record["index_bytes"],  # type: ignore[dict-item]
            "mapbox": record["mapbox_bytes"],  # type: ignore[dict-item]
        },
    )
    if result["transmitted"]:
        packet = result["packet"]  # type: ignore[index]
        append_packet(downlink_path, packet)  # type: ignore[arg-type]
        print(f"[{label}] transmitted  change={packet['change_type']}  packet={packet['packet_id']}")
        return True, "transmitted"
    print(f"[{label}] suppressed  reason={result['reason']}")
    return False, str(result["reason"])


def _run_loop(
    *,
    build_record,
    predict: PredictFn,
    downlink_path: Path,
    interval_seconds: int,
    labels: str,
) -> None:
    transmitted = 0
    suppressed = 0
    print(f"Loop mode: {labels}  |  interval: {interval_seconds}s  |  Ctrl+C to stop")
    while True:
        started = time.time()
        try:
            record = build_record()
            did_transmit, _reason = _process_record(record, predict=predict, downlink_path=downlink_path)
            if did_transmit:
                transmitted += 1
            else:
                suppressed += 1
        except SimSatNoImageError as exc:
            try:
                state = get_current_state()
                pos = f"lon={state['lon']:.2f} lat={state['lat']:.2f} t={state['timestamp']}"
            except Exception:
                pos = "position unavailable"
            cloud = exc.metadata.get("cloud_cover")
            if cloud is not None:
                print(f"[loop] cloud cover {cloud}% exceeds limit, skipping  ({pos})")
            else:
                print(f"[loop] no image for this step, skipping  ({pos})")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[loop] ERROR: {exc}")
        elapsed = time.time() - started
        sleep_seconds = max(0.0, interval_seconds - elapsed)
        print(f"[loop] totals  transmitted={transmitted} suppressed={suppressed}")
        time.sleep(sleep_seconds)


def _build_predictor(args: argparse.Namespace) -> tuple[PredictFn, object | None]:
    if args.backend == "anthropic":
        return anthropic_backend(), None
    if args.backend == "gemini":
        return gemini_backend(args.model or "gemini-3-flash-preview"), None
    if args.backend == "hf":
        if not args.model:
            print("--model is required for --backend hf")
            sys.exit(1)
        print(f"Loading Hugging Face checkpoint from {args.model} ...")
        return transformers_backend(args.model), None
    if not args.model:
        print("--model is required for --backend local")
        sys.exit(1)
    if not shutil.which("llama-server"):
        print("llama-server not found on PATH.")
        sys.exit(1)
    port = _choose_local_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is busy; using port {port} for llama-server.")
    print(f"Starting llama-server with model {args.model} on port {port} ...")
    server = start_llama_server(
        args.model,
        quant=args.quant or None,
        port=port,
        verbose=args.verbose_server,
        mmproj=args.mmproj,
    )
    try:
        wait_for_server(port=port)
    except TimeoutError as exc:
        print(str(exc))
        stop_server(server)
        sys.exit(1)
    print("llama-server ready.")
    return llama_backend(args.model, port), server


def main() -> None:
    parser = argparse.ArgumentParser(description="Satellite-side change suppression pipeline.")
    parser.add_argument("--backend", required=True, choices=["anthropic", "gemini", "local", "hf"])
    parser.add_argument("--model", default="", help="Model id for gemini/local/hf backends.")
    parser.add_argument("--quant", default="", help="Quant level for local llama-server backend.")
    parser.add_argument("--mmproj", default=None, help="Optional mmproj GGUF for local backend.")
    parser.add_argument("--port", type=int, default=8080, help="Preferred llama-server port.")
    parser.add_argument("--verbose-server", action="store_true", help="Show llama-server output.")
    parser.add_argument("--sample-dir", default=None, help="Saved sample directory to run once.")
    parser.add_argument("--dataset-dir", default=None, help="Replay a directory tree of saved sample folders.")
    parser.add_argument("--location", default=None, choices=list(LOCATIONS_BY_ID), help="Fetch a live tile for this location.")
    parser.add_argument("--timestamp", default=None, help="Timestamp for live SimSat fetch mode.")
    parser.add_argument("--current-loop", action="store_true", help="Continuously poll SimSat current position and process the live tile.")
    parser.add_argument("--watchlist-loop", action="store_true", help="Continuously cycle through the watchlist using the current SimSat timestamp.")
    parser.add_argument("--watchlist-location", default=None, choices=list(LOCATIONS_BY_ID), help="Restrict watchlist loop to one location.")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Loop interval for --current-loop or --watchlist-loop.")
    parser.add_argument("--size-km", type=float, default=5.0, help="Tile size for live fetch mode.")
    parser.add_argument("--downlink", default="downlink_packets.jsonl", help="JSONL packet output path.")
    args = parser.parse_args()

    mode_count = sum(
        bool(value)
        for value in (
            args.sample_dir,
            args.dataset_dir,
            args.location,
            args.current_loop,
            args.watchlist_loop,
        )
    )
    if mode_count != 1:
        print("Exactly one execution mode is required.")
        sys.exit(1)
    if args.location and not args.timestamp:
        print("--timestamp is required with --location.")
        sys.exit(1)

    predict, server = _build_predictor(args)
    downlink_path = Path(args.downlink)
    clear_cache()

    try:
        if args.current_loop:
            _run_loop(
                build_record=lambda: _current_record(args.size_km),
                predict=predict,
                downlink_path=downlink_path,
                interval_seconds=args.interval_seconds,
                labels="current-position live loop",
            )
            return

        if args.watchlist_loop:
            watch_ids = [args.watchlist_location] if args.watchlist_location else list(LOCATIONS_BY_ID)
            state = {"index": 0}

            def next_watch_record() -> dict[str, object]:
                current_state = get_current_state()
                if not current_state_is_ready(current_state):
                    raise ValueError(
                        "SimSat current simulation is not ready: current timestamp/position is still the zero-state."
                    )
                timestamp = str(current_state.get("timestamp") or _utc_now())
                location_id = str(watch_ids[state["index"] % len(watch_ids)])
                state["index"] += 1
                return _watch_record(location_id, timestamp, args.size_km)

            _run_loop(
                build_record=next_watch_record,
                predict=predict,
                downlink_path=downlink_path,
                interval_seconds=args.interval_seconds,
                labels=f"watchlist loop ({len(watch_ids)} locations)",
            )
            return

        if args.sample_dir:
            records = [_sample_record(Path(args.sample_dir))]
        elif args.dataset_dir:
            records = _iter_sample_records(Path(args.dataset_dir))
        else:
            records = [_fetch_live_record(str(args.location), str(args.timestamp), args.size_km)]

        transmitted = 0
        suppressed = 0
        for record in records:
            did_transmit, _reason = _process_record(record, predict=predict, downlink_path=downlink_path)
            if did_transmit:
                transmitted += 1
            else:
                suppressed += 1
        print(f"Done: transmitted={transmitted} suppressed={suppressed} downlink={downlink_path}")
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    main()

"""Generate wildfire risk annotations across a spatial and temporal tile grid.

Each run creates a timestamped folder under data/ with train/ and test/ splits,
e.g.:
    data/20260416_143052/train/attica_greece/s00_t00/rgb.png
    data/20260416_143052/train/attica_greece/s00_t00/swir.png
    data/20260416_143052/train/attica_greece/s00_t00/annotation.json
    data/20260416_143052/test/attica_greece/s00_t04/rgb.png
    ...

The split is determined by location: all tiles from reserved test locations go
to test/, and all remaining locations go to train/.

When --hf-dataset is passed the run directory is also pushed to Hugging Face Hub
as a dataset in leap-finetune VLM SFT format with train.jsonl and test.jsonl.

Usage:
    uv run scripts/generate_samples.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --n-temporal-tiles 12 --n-spatial-tiles 4 \\
        --test-locations attica_greece,mati_attica_gr

    uv run scripts/generate_samples.py \\
        --start-date 2024-06-01 --end-date 2024-09-01 \\
        --n-temporal-tiles 6 \\
        --location attica_greece

    uv run scripts/generate_samples.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --n-temporal-tiles 12 --n-spatial-tiles 4 \\
        --test-locations attica_greece,mati_attica_gr \\
        --dry-run
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

import requests
from tqdm import tqdm

from datacenter_watch.annotator import AnnotationParseError, annotate
from datacenter_watch.locations import LOCATIONS, LOCATIONS_BY_ID, Location
from datacenter_watch.modal_upload import (
    DEFAULT_MODAL_REMOTE_ROOT,
    DEFAULT_MODAL_VOLUME_NAME,
    upload_sample_dirs,
)
from datacenter_watch.simsat import (
    SimSatNoImageError,
    fetch_index_with_metadata,
    fetch_mapbox_with_metadata,
    fetch_rgb_with_metadata,
    fetch_swir_with_metadata,
)
from datacenter_watch.tiles import (
    TileCoord,
    spatial_grid,
    temporal_timestamps,
)

AnnotationResult: TypeAlias = dict[str, object]

DATA_DIR = Path(__file__).parent.parent / "data"

_RETRY_DELAYS = [5, 15, 30]  # seconds between retries on 429


@dataclass(frozen=True)
class TileTask:
    loc: Location
    spatial: TileCoord
    timestamp: str
    split: str          # "train" | "test"
    spatial_idx: int
    temporal_idx: int
    include_mapbox: bool

    @property
    def label(self) -> str:
        return f"{self.loc.id}/s{self.spatial_idx:02d}_t{self.temporal_idx:02d}"

    @property
    def tile_key(self) -> str:
        return f"s{self.spatial_idx:02d}_t{self.temporal_idx:02d}"


def _annotate_with_retry(
    rgb_bytes: bytes,
    swir_bytes: bytes,
    *,
    index_bytes: bytes | None,
    mapbox_bytes: bytes | None,
    model: str,
    provider: str,
) -> dict[str, object]:
    """Call annotate with retries for transient provider errors."""
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            return annotate(
                rgb_bytes,
                swir_bytes,
                index_bytes=index_bytes,
                mapbox_bytes=mapbox_bytes,
                model=model,
                provider=provider,
            )
        except Exception as exc:
            if _is_retryable_annotation_error(exc) and attempt <= len(_RETRY_DELAYS):
                tqdm.write(f"  transient model error, retrying in {delay}s ...")
                time.sleep(delay)
            else:
                raise
    return annotate(
        rgb_bytes,
        swir_bytes,
        index_bytes=index_bytes,
        mapbox_bytes=mapbox_bytes,
        model=model,
        provider=provider,
    )


def _is_retryable_annotation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in ("429", "503", "unavailable", "rate limit", "resource exhausted")
    )

def _summary_label(result: AnnotationResult) -> str:
    detections = result.get("detections")
    if isinstance(detections, list):
        return f"detections={len(detections)}"
    return "annotated"


def process_tile(
    task: TileTask,
    sample_dir: Path,
    size_km: float,
    dry_run: bool,
    model: str,
    provider: str,
) -> AnnotationResult | None:
    sample_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"[{task.label}] fetching images ...")
    index_bytes: bytes | None = None
    mapbox_bytes: bytes | None = None
    imagery_metadata: dict[str, object] = {
        "requested_timestamp": task.timestamp,
        "include_mapbox": task.include_mapbox,
    }
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rgb_future = pool.submit(
                fetch_rgb_with_metadata, task.spatial.lon, task.spatial.lat, task.timestamp, size_km
            )
            swir_future = pool.submit(
                fetch_swir_with_metadata, task.spatial.lon, task.spatial.lat, task.timestamp, size_km
            )
            index_future = pool.submit(
                fetch_index_with_metadata, task.spatial.lon, task.spatial.lat, task.timestamp, size_km
            )
            mapbox_future = None
            if task.include_mapbox:
                mapbox_future = pool.submit(
                    fetch_mapbox_with_metadata, task.spatial.lon, task.spatial.lat
                )
            rgb_bytes, rgb_meta = rgb_future.result()
            swir_bytes, swir_meta = swir_future.result()
            index_bytes, index_meta = index_future.result()
            imagery_metadata["rgb"] = rgb_meta
            imagery_metadata["swir"] = swir_meta
            imagery_metadata["index"] = index_meta
            if mapbox_future is not None:
                try:
                    mapbox_bytes, mapbox_meta = mapbox_future.result()
                    imagery_metadata["mapbox"] = mapbox_meta
                    if not mapbox_bytes:
                        mapbox_bytes = None
                except requests.HTTPError as exc:
                    tqdm.write(f"[{task.label}] mapbox unavailable: {exc.response.status_code}")
    except SimSatNoImageError as exc:
        actual_dt = exc.metadata.get("datetime")
        requested_ts = exc.metadata.get("timestamp", task.timestamp)
        if actual_dt:
            tqdm.write(
                f"[{task.label}] SKIP: no Sentinel image available"
                f"  requested={requested_ts}  nearest={actual_dt}"
            )
        else:
            tqdm.write(f"[{task.label}] SKIP: no Sentinel image available  requested={requested_ts}")
        return None
    except requests.HTTPError as exc:
        tqdm.write(f"[{task.label}] SKIP: SimSat returned {exc.response.status_code}")
        return None

    (sample_dir / "rgb.png").write_bytes(rgb_bytes)
    (sample_dir / "swir.png").write_bytes(swir_bytes)
    if index_bytes is not None:
        (sample_dir / "index.png").write_bytes(index_bytes)
    if mapbox_bytes:
        (sample_dir / "mapbox.png").write_bytes(mapbox_bytes)
    (sample_dir / "imagery_metadata.json").write_text(
        json.dumps(imagery_metadata, indent=2), encoding="utf-8"
    )

    if dry_run:
        tqdm.write(f"[{task.label}] dry-run: images saved, skipping annotation")
        return None

    tqdm.write(f"[{task.label}] annotating with {model} ...")
    try:
        result = _annotate_with_retry(
            rgb_bytes,
            swir_bytes,
            index_bytes=index_bytes,
            mapbox_bytes=mapbox_bytes,
            model=model,
            provider=provider,
        )
    except AnnotationParseError as exc:
        raw_path = sample_dir / "annotation_raw.txt"
        raw_path.write_text(exc.raw_response, encoding="utf-8")
        tqdm.write(f"[{task.label}] ERROR: {exc}  raw saved to {raw_path}")
        return None
    except ValueError as exc:
        tqdm.write(f"[{task.label}] ERROR: {exc}")
        return None

    annotation: AnnotationResult = {
        "id": task.loc.id,
        "split": task.split,
        "spatial_index": task.spatial_idx,
        "temporal_index": task.temporal_idx,
        "lon": task.spatial.lon,
        "lat": task.spatial.lat,
        "timestamp": task.timestamp,
        "size_km": size_km,
        "annotator_model": model,
        "annotator_provider": provider,
        **result,
    }
    (sample_dir / "annotation.json").write_text(
        json.dumps(annotation, indent=2), encoding="utf-8"
    )
    tqdm.write(f"[{task.label}] done  split={task.split}  {_summary_label(result)}")
    return annotation


def _parse_test_locations(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _deterministic_test_locations(locations: list[Location], test_ratio: float) -> set[str]:
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"--test-ratio must be between 0 and 1, got {test_ratio}")
    if test_ratio == 0.0:
        return set()
    if test_ratio == 1.0:
        return {loc.id for loc in locations}

    scored = sorted(
        (
            int.from_bytes(
                hashlib.sha256(loc.id.encode("utf-8")).digest()[:8],
                byteorder="big",
                signed=False,
            ),
            loc.id,
        )
        for loc in locations
    )
    n_test = round(len(locations) * test_ratio)
    n_test = max(1, min(len(locations), n_test))
    return {loc_id for _, loc_id in scored[:n_test]}


def _flush_modal_batch(
    run_dir: Path,
    pending_sample_dirs: list[Path],
    *,
    volume_name: str,
    remote_root: str,
) -> list[Path]:
    if not pending_sample_dirs:
        return pending_sample_dirs
    try:
        uploaded = upload_sample_dirs(
            run_dir,
            pending_sample_dirs,
            volume_name=volume_name,
            remote_root=remote_root,
        )
        print(
            f"Uploaded {uploaded} samples to Modal volume '{volume_name}'"
            f" under {remote_root.rstrip('/')}/{run_dir.name}",
            flush=True,
        )
        return []
    except Exception as exc:
        print(f"Modal upload failed; will retry later: {exc}", flush=True)
        return pending_sample_dirs


def push_to_hf(
    run_dir: Path,
    results: list[AnnotationResult],
    dataset_name: str,
) -> None:
    """Push annotations to Hugging Face Hub as a flat tabular dataset.

    Creates a parquet dataset with string-only columns so HF never embeds
    images as binary blobs. Image files are uploaded separately to the images/
    subdirectory of the same repo.

    Columns: region, timestamp, split, rgb_path, swir_path, index_path,
    mapbox_path, output.
    Image paths are relative to the repo root (e.g. images/foo_rgb.png).
    output is the JSON-serialised model annotation.

    To convert this dataset to leap-finetune JSONL format, run prepare_datacenter_watch.py.

    Auth is handled via the HF_TOKEN environment variable.
    """
    from datasets import Dataset, DatasetDict, Features, Value
    from huggingface_hub import HfApi

    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    for ann in results:
        loc_id = str(ann["id"])
        si = int(ann["spatial_index"])  # type: ignore[arg-type]
        ti = int(ann["temporal_index"])  # type: ignore[arg-type]
        tile_key = f"{loc_id}_s{si:02d}_t{ti:02d}"

        rgb_name = f"{tile_key}_rgb.png"
        swir_name = f"{tile_key}_swir.png"
        split = str(ann["split"])
        tile_dir = run_dir / split / loc_id / f"s{si:02d}_t{ti:02d}"
        shutil.copy2(tile_dir / "rgb.png", images_dir / rgb_name)
        shutil.copy2(tile_dir / "swir.png", images_dir / swir_name)
        index_name = ""
        mapbox_name = ""
        index_path = tile_dir / "index.png"
        mapbox_path = tile_dir / "mapbox.png"
        if index_path.exists():
            index_name = f"{tile_key}_index.png"
            shutil.copy2(index_path, images_dir / index_name)
        if mapbox_path.exists():
            mapbox_name = f"{tile_key}_mapbox.png"
            shutil.copy2(mapbox_path, images_dir / mapbox_name)

        rows.append({
            "region":    loc_id,
            "timestamp": str(ann["timestamp"]),
            "split":     split,
            "rgb_path":  f"images/{rgb_name}",
            "swir_path": f"images/{swir_name}",
            "index_path": f"images/{index_name}" if index_name else "",
            "mapbox_path": f"images/{mapbox_name}" if mapbox_name else "",
            "output":    json.dumps({
                k: v
                for k, v in ann.items()
                if k not in {
                    "id", "split", "spatial_index", "temporal_index",
                    "lon", "lat", "timestamp", "size_km", "annotator_model", "annotator_provider",
                }
            }),
        })

    features = Features({
        "region":    Value("string"),
        "timestamp": Value("string"),
        "split":     Value("string"),
        "rgb_path":  Value("string"),
        "swir_path": Value("string"),
        "index_path": Value("string"),
        "mapbox_path": Value("string"),
        "output":    Value("string"),
    })

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows  = [r for r in rows if r["split"] == "test"]
    ds_dict: dict[str, Dataset] = {"train": Dataset.from_list(train_rows, features=features)}
    if test_rows:
        ds_dict["test"] = Dataset.from_list(test_rows, features=features)

    api = HfApi()
    api.create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True)
    DatasetDict(ds_dict).push_to_hub(dataset_name)
    print(f"  train: {len(train_rows)} rows  test: {len(test_rows)} rows")

    api.upload_folder(
        folder_path=str(images_dir),
        path_in_repo="images",
        repo_id=dataset_name,
        repo_type="dataset",
    )
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wildfire risk annotations across a spatial and temporal tile grid."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        metavar="DATE",
        help="Start of the sampling window, ISO 8601 date, e.g. 2024-01-01.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        metavar="DATE",
        help="End of the sampling window, ISO 8601 date, e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--n-temporal-tiles",
        type=int,
        default=1,
        metavar="N",
        help="Number of timestamps to sample per location within the window (default: 1).",
    )
    parser.add_argument(
        "--n-spatial-tiles",
        type=int,
        default=1,
        metavar="N",
        help="Number of spatial grid tiles per location per timestamp (default: 1).",
    )
    parser.add_argument(
        "--test-locations",
        default="",
        metavar="IDS",
        help=(
            "Comma-separated location ids reserved for the test split, "
            "e.g. attica_greece,mati_attica_gr. All other locations go to train."
        ),
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        metavar="R",
        help=(
            "Deterministically reserve this fraction of loaded locations for test. "
            "Ignored if --test-locations is also provided."
        ),
    )
    parser.add_argument(
        "--size-km",
        type=float,
        default=5.0,
        metavar="KM",
        help="Tile edge length in km, also used as the spatial grid spacing (default: 5.0).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Number of tiles to annotate in parallel (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch images but skip the Opus annotation call.",
    )
    parser.add_argument(
        "--location",
        metavar="ID",
        help="Process a single location by its id (e.g. attica_greece).",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        metavar="MODEL",
        help="Anthropic model ID to use for annotation (default: claude-opus-4-6).",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "anthropic", "gemini"],
        default="auto",
        help="Annotation provider. Use 'gemini' for Gemini models such as gemini-3.0-flash.",
    )
    parser.add_argument(
        "--hf-dataset",
        metavar="REPO",
        default=None,
        help=(
            "Hugging Face dataset repo to push results to, e.g. username/wildfire-risk."
            " Requires HF_TOKEN env var. Skipped if not provided."
        ),
    )
    parser.add_argument(
        "--modal-upload",
        action="store_true",
        help="Upload annotated sample directories to a Modal volume during the run.",
    )
    parser.add_argument(
        "--modal-batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Upload to Modal every N completed annotations (default: 50).",
    )
    parser.add_argument(
        "--modal-volume",
        default=DEFAULT_MODAL_VOLUME_NAME,
        metavar="NAME",
        help=f"Modal volume name (default: {DEFAULT_MODAL_VOLUME_NAME}).",
    )
    parser.add_argument(
        "--modal-remote-root",
        default=DEFAULT_MODAL_REMOTE_ROOT,
        metavar="PATH",
        help=f"Remote root path inside the Modal volume (default: {DEFAULT_MODAL_REMOTE_ROOT}).",
    )
    args = parser.parse_args()

    # Parse and validate dates.
    try:
        start_dt = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        print(f"Invalid date: {exc}")
        sys.exit(1)
    if end_dt <= start_dt:
        print("--end-date must be after --start-date")
        sys.exit(1)

    # Resolve locations.
    locations = list(LOCATIONS)
    if args.location:
        if args.location not in LOCATIONS_BY_ID:
            ids = ", ".join(loc.id for loc in LOCATIONS)
            print(f"Unknown location id '{args.location}'. Available: {ids}")
            sys.exit(1)
        locations = [LOCATIONS_BY_ID[args.location]]

    test_locations = _parse_test_locations(args.test_locations)
    if args.test_ratio is not None:
        if test_locations:
            print("--test-ratio is ignored when --test-locations is provided")
        else:
            test_locations = _deterministic_test_locations(locations, args.test_ratio)
    unknown_test_locations = sorted(test_locations - set(LOCATIONS_BY_ID))
    if unknown_test_locations:
        print(
            "Unknown test location ids: "
            + ", ".join(unknown_test_locations)
        )
        sys.exit(1)

    # Build tile grid.
    timestamps = temporal_timestamps(start_dt, end_dt, args.n_temporal_tiles)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(locations) * args.n_temporal_tiles * args.n_spatial_tiles
    print(
        f"Run: {run_id}"
        f"  |  provider: {args.provider}"
        f"  |  model: {args.model}"
        f"  |  locations: {len(locations)}"
        f"  |  temporal_tiles: {args.n_temporal_tiles}"
        f"  |  spatial_tiles: {args.n_spatial_tiles}"
        f"  |  total: {n_total}"
        f"  |  test_locations: {','.join(sorted(test_locations)) or 'none'}"
        f"  |  concurrency: {args.concurrency}"
    )

    # Build (task, sample_dir) pairs.
    task_pairs: list[tuple[TileTask, Path]] = []
    for loc in locations:
        spatial_tiles = spatial_grid(loc.lon, loc.lat, args.n_spatial_tiles, args.size_km)
        for ti, ts in enumerate(timestamps):
            split = "test" if loc.id in test_locations else "train"
            for si, spatial in enumerate(spatial_tiles):
                task = TileTask(
                    loc=loc,
                    spatial=spatial,
                    timestamp=ts,
                    split=split,
                    spatial_idx=si,
                    temporal_idx=ti,
                    include_mapbox=ti == len(timestamps) - 1,
                )
                sample_dir = run_dir / split / loc.id / task.tile_key
                task_pairs.append((task, sample_dir))

    # Run in parallel.
    annotations: list[AnnotationResult] = []
    pending_modal_uploads: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_tile, task, sample_dir, args.size_km, args.dry_run, args.model, args.provider): (task, sample_dir)
            for task, sample_dir in task_pairs
        }
        with tqdm(total=len(task_pairs), desc="tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                task, sample_dir = futures[future]
                exc = future.exception()
                if exc:
                    tqdm.write(f"[{task.label}] UNEXPECTED ERROR: {exc}")
                else:
                    result = future.result()
                    if result is not None:
                        annotations.append(result)
                        if args.modal_upload:
                            pending_modal_uploads.append(sample_dir)
                            if len(pending_modal_uploads) >= args.modal_batch_size:
                                pending_modal_uploads = _flush_modal_batch(
                                    run_dir,
                                    pending_modal_uploads,
                                    volume_name=args.modal_volume,
                                    remote_root=args.modal_remote_root,
                                )
                        pbar.set_postfix(
                            split=task.split,
                            detections=len(result.get("detections", [])) if isinstance(result.get("detections"), list) else "?",
                        )
                pbar.update(1)

    train_count = sum(1 for a in annotations if a["split"] == "train")
    test_count = sum(1 for a in annotations if a["split"] == "test")
    print(f"\nDone: {len(annotations)} annotations  (train={train_count}  test={test_count})")

    train_location_ids = sorted({str(a["id"]) for a in annotations if a.get("split") == "train"})
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "test_ratio": args.test_ratio,
                "test_locations": sorted(test_locations),
                "train_locations": train_location_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.modal_upload:
        pending_modal_uploads = _flush_modal_batch(
            run_dir,
            pending_modal_uploads,
            volume_name=args.modal_volume,
            remote_root=args.modal_remote_root,
        )
        if pending_modal_uploads:
            print(
                f"Modal upload still pending for {len(pending_modal_uploads)} samples after final flush.",
                flush=True,
            )

    if args.hf_dataset:
        if not annotations:
            print("No annotations produced; skipping Hugging Face push.")
        else:
            print(f"Pushing {len(annotations)} samples to {args.hf_dataset} ...")
            push_to_hf(run_dir, annotations, args.hf_dataset)


if __name__ == "__main__":
    main()

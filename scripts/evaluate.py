"""Run an evaluation against a generated dataset.

--split is required: use 'test' to evaluate model quality, 'train' to check Opus self-consistency.

Usage:
    uv run scripts/evaluate.py --hf-dataset Paulescu/datacenter_watch --backend anthropic --split test
    uv run scripts/evaluate.py --hf-dataset Paulescu/datacenter_watch --backend gemini --split test
    uv run scripts/evaluate.py --hf-dataset Paulescu/datacenter_watch --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --split test
    uv run scripts/evaluate.py --dataset data/20260421_150039 --backend anthropic --split test
"""

import argparse
import json
import shutil
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

from huggingface_hub import snapshot_download

from datacenter_watch.annotator import annotate_raw
from datacenter_watch.compact_schema import migrate_annotation
from datacenter_watch.evaluator import (
    EVAL_FIELDS,
    EvalSummary,
    SampleResult,
    anthropic_backend,
    evaluate_sample,
    gemini_backend,
    llama_backend,
    model_name,
    render_report,
    save_results,
    start_llama_server,
    stop_server,
    transformers_backend,
    wait_for_server,
)

EVALS_DIR = Path(__file__).parent.parent / "evals"

# (sample_id, rgb_bytes, swir_bytes, index_bytes, mapbox_bytes, ground_truth)
SampleData: TypeAlias = tuple[
    str,
    bytes,
    bytes,
    bytes | None,
    bytes | None,
    dict[str, object],
]


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _choose_local_port(requested_port: int) -> int:
    if _port_is_available(requested_port):
        return requested_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_local_samples(dataset_dir: Path, split: str) -> list[SampleData]:
    """Load samples from a local run directory (train|test/{loc}/{tile}/ layout)."""
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        print(f"Split '{split}' not found in {dataset_dir}")
        sys.exit(1)

    samples: list[SampleData] = []
    for loc_dir in sorted(split_dir.iterdir()):
        if not loc_dir.is_dir():
            continue
        for tile_dir in sorted(loc_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            sample_id = f"{loc_dir.name}/{tile_dir.name}"
            rgb_path = tile_dir / "rgb.png"
            swir_path = tile_dir / "swir.png"
            index_path = tile_dir / "index.png"
            mapbox_path = tile_dir / "mapbox.png"
            annotation_path = tile_dir / "annotation.json"
            if not (rgb_path.exists() and swir_path.exists() and annotation_path.exists()):
                print(f"[{sample_id}] SKIP: missing files")
                continue
            ground_truth = migrate_annotation(
                json.loads(annotation_path.read_text(encoding="utf-8"))
            )
            samples.append((
                sample_id,
                rgb_path.read_bytes(),
                swir_path.read_bytes(),
                index_path.read_bytes() if index_path.exists() else None,
                mapbox_path.read_bytes() if mapbox_path.exists() else None,
                ground_truth,
            ))
    return samples


def load_hf_samples(snapshot_dir: Path, split: str) -> list[SampleData]:
    """Load samples from a HF snapshot (parquet + flat images/ layout)."""
    from datasets import load_dataset

    ds = load_dataset(str(snapshot_dir), split=split)
    samples: list[SampleData] = []
    for row in ds:
        region    = str(row["region"])
        rgb_path  = snapshot_dir / str(row["rgb_path"])
        swir_path = snapshot_dir / str(row["swir_path"])
        index_path = snapshot_dir / str(row["index_path"]) if str(row.get("index_path", "")) else None
        mapbox_path = snapshot_dir / str(row["mapbox_path"]) if str(row.get("mapbox_path", "")) else None
        # derive tile key from filename: e.g. attica_greece_s00_t00_rgb.png → s00_t00
        tile_key  = Path(str(row["rgb_path"])).stem.removesuffix("_rgb")[len(region) + 1:]
        sample_id = f"{region}/{tile_key}"
        ground_truth = migrate_annotation(json.loads(str(row["output"])))
        samples.append((
            sample_id,
            rgb_path.read_bytes(),
            swir_path.read_bytes(),
            index_path.read_bytes() if index_path and index_path.exists() else None,
            mapbox_path.read_bytes() if mapbox_path and mapbox_path.exists() else None,
            ground_truth,
        ))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wildfire risk predictions.")
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        help="Path to a local dataset run, e.g. data/20260421_150039.",
    )
    parser.add_argument(
        "--hf-dataset",
        metavar="REPO",
        help="Hugging Face dataset repo to evaluate against, e.g. Paulescu/datacenter_watch. Downloads via snapshot_download and uses the cached copy.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["anthropic", "gemini", "local", "hf"],
        help="Inference backend: 'anthropic' (Opus API), 'gemini' (Gemini API), 'local' (llama-server GGUF), 'hf' (HuggingFace safetensors checkpoint).",
    )
    parser.add_argument(
        "--model",
        metavar="REPO",
        default="",
        help=(
            "Model ID to evaluate with. Required for --backend local and hf; "
            "defaults to gemini-3-flash-preview for --backend gemini."
        ),
    )
    parser.add_argument(
        "--quant",
        metavar="QUANT",
        default="",
        help="Quantization level within the repo (e.g. Q8_0). Appended as <repo>:<quant>.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="llama-server port (default: 8080, local backend only).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallel workers (default: 3 for anthropic, 1 for local).",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test"],
        help="Which data split to evaluate: 'train' checks Opus self-consistency, 'test' evaluates model quality.",
    )
    parser.add_argument(
        "--mmproj",
        metavar="PATH",
        default=None,
        help="Path to the mmproj GGUF (vision tower + projector). Required for VLM inference with a local fine-tuned GGUF.",
    )
    parser.add_argument(
        "--verbose-server",
        action="store_true",
        help="Show llama-server output (local backend only).",
    )
    parser.add_argument(
        "--dump-raw-location",
        default="",
        metavar="ID",
        help="If set, dump raw responses for samples whose id starts with this location id.",
    )
    parser.add_argument(
        "--dump-raw-dir",
        default=None,
        metavar="DIR",
        help="Directory to write raw responses into (default: evals/<run_id>/raw).",
    )
    args = parser.parse_args()

    if not args.dataset and not args.hf_dataset:
        print("Either --dataset or --hf-dataset is required.")
        sys.exit(1)
    if args.dataset and args.hf_dataset:
        print("--dataset and --hf-dataset are mutually exclusive.")
        sys.exit(1)

    if args.backend in ("local", "hf") and not args.model:
        print("--model is required when using --backend local or hf")
        sys.exit(1)

    if args.backend == "local" and not shutil.which("llama-server"):
        print("llama-server not found on PATH. Install llama.cpp and ensure llama-server is available.")
        sys.exit(1)

    if args.hf_dataset:
        print(f"Downloading dataset from Hugging Face: {args.hf_dataset} ...")
        snapshot_dir = Path(snapshot_download(repo_id=args.hf_dataset, repo_type="dataset"))
        print(f"Snapshot at {snapshot_dir}")
        samples = load_hf_samples(snapshot_dir, args.split)
        dataset_label = args.hf_dataset
    else:
        local_dir = Path(args.dataset)
        if not local_dir.is_dir():
            print(f"Dataset not found: {local_dir}")
            sys.exit(1)
        samples = load_local_samples(local_dir, args.split)
        dataset_label = args.dataset

    if not samples:
        print(f"No samples found for split '{args.split}'.")
        sys.exit(1)

    concurrency = args.concurrency or (1 if args.backend == "local" else 3)
    eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"Eval: {eval_run_id}  |  dataset: {dataset_label}  |  split: {args.split}"
        f"  |  samples: {len(samples)}  |  backend: {args.backend}"
        f"  |  concurrency: {concurrency}"
    )

    raw_dump_dir = Path(args.dump_raw_dir) if args.dump_raw_dir else EVALS_DIR / eval_run_id / "raw"
    if args.dump_raw_location:
        raw_dump_dir.mkdir(parents=True, exist_ok=True)

    # Start llama-server if needed.
    server_process = None
    llama_port = args.port
    if args.backend == "local":
        llama_port = _choose_local_port(args.port)
        if llama_port != args.port:
            print(f"Port {args.port} is busy; using port {llama_port} for llama-server.")
        print(f"Starting llama-server with model {args.model} on port {llama_port} ...")
        server_process = start_llama_server(
            args.model,
            quant=args.quant or None,
            port=llama_port,
            verbose=args.verbose_server,
            mmproj=args.mmproj,
        )
        try:
            wait_for_server(port=llama_port)
        except TimeoutError as exc:
            print(str(exc))
            stop_server(server_process)
            sys.exit(1)
        print("llama-server ready.")

    if args.backend == "anthropic":
        predict = anthropic_backend()
    elif args.backend == "gemini":
        predict = gemini_backend(args.model or "gemini-3-flash-preview")
    elif args.backend == "hf":
        print(f"Loading HuggingFace checkpoint from {args.model} ...")
        predict = transformers_backend(args.model)
        print("Model loaded.")
    else:
        predict = llama_backend(args.model, llama_port)

    results: list[SampleResult] = []
    sample_order = {sid: i for i, (sid, *_) in enumerate(samples)}
    sample_lookup = {sid: (rgb, swir, index, mapbox) for sid, rgb, swir, index, mapbox, _gt in samples}
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(evaluate_sample, sid, rgb, swir, index, mapbox, gt, predict): sid
                for sid, rgb, swir, index, mapbox, gt in samples
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                fm = result.field_matches
                status = " ".join(
                    f"{f[:4]}={'✓' if fm.get(f) else '✗'}" for f in EVAL_FIELDS
                )
                print(f"[{result.id}] {status}", flush=True)
                if args.dump_raw_location is not None and result.id.startswith(f"{args.dump_raw_location}"):
                    raw_path = raw_dump_dir / result.id / "raw.txt"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_data = {
                        "prediction": result.prediction,
                        "ground_truth": result.ground_truth,
                        "field_matches": result.field_matches,
                    }
                    raw_path.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")
    finally:
        if server_process is not None:
            stop_server(server_process)

    results.sort(key=lambda r: sample_order.get(r.id, 999))

    summary = EvalSummary(results=results)
    mname = model_name(args.backend, args.model, args.quant)
    report = render_report(summary, f"{dataset_label}/{args.split}", args.backend, mname, eval_run_id)

    eval_dir = EVALS_DIR / eval_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.md").write_text(report, encoding="utf-8")
    save_results(
        eval_dir,
        summary,
        dataset=f"{dataset_label}/{args.split}",
        backend=args.backend,
        model=mname,
        split=args.split,
        eval_run_id=eval_run_id,
    )

    print()
    print(report)
    print(f"Report saved to evals/{eval_run_id}/report.md")
    print(f"Results saved to evals/{eval_run_id}/results.json")


if __name__ == "__main__":
    main()

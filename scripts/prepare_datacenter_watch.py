"""Convert the Paulescu/datacenter_watch HF dataset to leap-finetune VLM SFT format.

Downloads the dataset from HuggingFace Hub and writes
datacenter_watch_train.jsonl and datacenter_watch_test.jsonl in the
leap-finetune messages format. Images are already in the snapshot and do not
need to be copied.

The output directory will contain:
    datacenter_watch_train.jsonl    -- training samples (messages format)
    datacenter_watch_test.jsonl     -- evaluation samples (messages format)

The images/ directory lives inside the HF snapshot (resolved automatically by
huggingface_hub.snapshot_download). The leap-finetune config must set
image_root to <output_dir>/images/.

Usage:
    uv run scripts/prepare_datacenter_watch.py
    uv run scripts/prepare_datacenter_watch.py --dataset Paulescu/datacenter_watch --output ./data/datacenter_watch
    uv run scripts/prepare_datacenter_watch.py --dataset Paulescu/datacenter_watch --modal
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from datacenter_watch.annotator import SYSTEM_PROMPT, build_user_text
from datacenter_watch.compact_schema import migrate_annotation

DEFAULT_DATASET = "Paulescu/datacenter_watch"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "datacenter_watch"

# Modal configuration
MODAL_VOLUME_NAME = "datacenter_watch"
MODAL_MOUNT_POINT = "/datacenter_watch"
MODAL_OUTPUT_DIR = f"{MODAL_MOUNT_POINT}/data/datacenter_watch"


def make_vlm_row(
    rgb_name: str,
    swir_name: str,
    output: str,
    *,
    index_name: str = "",
    mapbox_name: str = "",
) -> dict[str, object]:
    """Build one leap-finetune VLM SFT row from image filenames and model output."""
    content: list[dict[str, str]] = [
        {"type": "image", "image": rgb_name},
        {"type": "image", "image": swir_name},
    ]
    if index_name:
        content.append({"type": "image", "image": index_name})
    if mapbox_name:
        content.append({"type": "image", "image": mapbox_name})
    content.append({
        "type": "text",
        "text": f"{SYSTEM_PROMPT.strip()}\n\n{build_user_text(bool(index_name), bool(mapbox_name))}",
    })
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
            },
        ]
    }


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"  Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert datacenter_watch HF dataset to leap-finetune JSONL format."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        metavar="REPO",
        help=f"HuggingFace dataset repo (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="DIR",
        help=f"Directory to write JSONL files (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help=(
            f"Run data preparation on Modal (serverless cloud). "
            f"Writes output to the Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_MOUNT_POINT}/. "
        ),
    )
    args = parser.parse_args()

    if args.modal:
        _run_on_modal(args)
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the snapshot directly into output_dir so images land at
    # output_dir/images/ with no intermediate copy (avoids a slow per-file
    # network copy when output_dir lives on a Modal volume).
    print(f"Downloading snapshot of {args.dataset} into {output_dir} ...")
    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    print(f"  Download complete.")

    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"images/ directory not found at {images_dir}. "
            "Re-run generate_samples.py --hf-dataset to regenerate the dataset."
        )
    print(f"  {sum(1 for _ in images_dir.iterdir())} images available.")

    print(f"Loading dataset from {output_dir} ...")
    ds = load_dataset(str(output_dir))

    for split_name in ("train", "test"):
        if split_name not in ds:
            print(f"  Split '{split_name}' not found, skipping.")
            continue

        rows: list[dict[str, object]] = []
        for row in ds[split_name]:
            rgb_name  = Path(str(row["rgb_path"])).name
            swir_name = Path(str(row["swir_path"])).name
            index_name = Path(str(row["index_path"])).name if str(row.get("index_path", "")) else ""
            mapbox_name = Path(str(row["mapbox_path"])).name if str(row.get("mapbox_path", "")) else ""
            migrated_output = migrate_annotation(json.loads(str(row["output"])))
            output = json.dumps(migrated_output, separators=(",", ":"))
            rows.append(
                make_vlm_row(
                    rgb_name,
                    swir_name,
                    output,
                    index_name=index_name,
                    mapbox_name=mapbox_name,
                )
            )

        write_jsonl(rows, output_dir / f"datacenter_watch_{split_name}.jsonl")

    print(f"\nDone. Set image_root to: {images_dir}")
    print("Training config: uv run leap-finetune configs/datacenter_watch_finetune.yaml")


def _run_on_modal(args: argparse.Namespace) -> None:
    """Run the data preparation pipeline on Modal (no local disk or bandwidth required)."""
    import modal

    app = modal.App("datacenter_watch-data-prep")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    src_dir = Path(__file__).parent.parent / "src" / "datacenter_watch"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("datasets", "huggingface_hub", "anthropic")
        .add_local_file(__file__, "/app/prepare_datacenter_watch.py", copy=True)
        .add_local_dir(str(src_dir), "/app/datacenter_watch", copy=True)
    )

    @app.function(
        image=image,
        volumes={MODAL_MOUNT_POINT: volume},
        timeout=3600,
        serialized=True,
        secrets=[modal.Secret.from_local_environ(env_keys=["HF_TOKEN"])],
    )
    def prepare(dataset: str, output: str) -> None:
        cmd = [
            sys.executable,
            "/app/prepare_datacenter_watch.py",
            "--dataset", dataset,
            "--output", output,
        ]
        env = {**os.environ, "PYTHONPATH": "/app"}
        subprocess.run(cmd, check=True, env=env)
        volume.commit()

    print(f"Preparing datacenter_watch dataset on Modal (volume: '{MODAL_VOLUME_NAME}')...")
    with modal.enable_output():
        with app.run():
            prepare.remote(args.dataset, MODAL_OUTPUT_DIR)

    print(f"\nData ready in Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_OUTPUT_DIR}.")
    print(f"Set image_root to: {MODAL_OUTPUT_DIR}/images/")
    print("Next step: uv run leap-finetune configs/datacenter_watch_finetune_modal.yaml")


if __name__ == "__main__":
    main()

"""Upload generated sample directories to a Modal volume.

Usage:
    uv run scripts/upload_to_modal.py --run-dir data/20260506_120401
"""

import argparse
from pathlib import Path

from datacenter_watch.modal_upload import (
    DEFAULT_MODAL_REMOTE_ROOT,
    DEFAULT_MODAL_VOLUME_NAME,
    iter_annotated_sample_dirs,
    upload_sample_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a generated run to a Modal volume.")
    parser.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Path to a local run directory, e.g. data/20260506_120401.",
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
        help=f"Remote root path inside the volume (default: {DEFAULT_MODAL_REMOTE_ROOT}).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sample_dirs = iter_annotated_sample_dirs(run_dir)
    uploaded = upload_sample_dirs(
        run_dir,
        sample_dirs,
        volume_name=args.modal_volume,
        remote_root=args.modal_remote_root,
    )
    print(
        f"Uploaded {uploaded} sample directories to volume '{args.modal_volume}'"
        f" under {args.modal_remote_root.rstrip('/')}/{run_dir.name}"
    )


if __name__ == "__main__":
    main()

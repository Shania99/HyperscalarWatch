from pathlib import Path

import modal

DEFAULT_MODAL_VOLUME_NAME = "datacenter_watch-data"
DEFAULT_MODAL_REMOTE_ROOT = "/runs"


def iter_annotated_sample_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in run_dir.glob("*/*/s*_t*/annotation.json")
    )


def upload_sample_dirs(
    run_dir: Path,
    sample_dirs: list[Path],
    *,
    volume_name: str = DEFAULT_MODAL_VOLUME_NAME,
    remote_root: str = DEFAULT_MODAL_REMOTE_ROOT,
    force: bool = True,
) -> int:
    if not sample_dirs:
        return 0

    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    remote_root = remote_root.rstrip("/") or "/"

    with volume.batch_upload(force=force) as batch:
        for sample_dir in sample_dirs:
            relative = sample_dir.relative_to(run_dir)
            remote_dir = f"{remote_root}/{run_dir.name}/{relative.as_posix()}"
            batch.put_directory(str(sample_dir), remote_dir)
    return len(sample_dirs)

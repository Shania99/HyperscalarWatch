"""Ground-side ingest for downlinked satellite packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from datacenter_watch.ground_ops import ingest_packet, init_ground_db


def _ingest_available(
    conn,
    *,
    downlink_path: Path,
    offset: int,
) -> tuple[int, int, int, int, int]:
    processed = 0
    skipped = 0
    new_sites = 0
    updated_sites = 0
    alerts = 0
    with downlink_path.open(encoding="utf-8") as fh:
        fh.seek(offset)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            packet = json.loads(line)
            result = ingest_packet(conn, packet)
            if not result["processed"]:
                skipped += 1
                continue
            processed += 1
            new_sites += int(result["new_sites"])
            updated_sites += int(result["updated_sites"])
            alerts += int(result["alerts"])
            print(
                f"[{packet['packet_id']}] processed"
                f" new_sites={result['new_sites']}"
                f" updated_sites={result['updated_sites']}"
                f" alerts={result['alerts']}"
            )
        new_offset = fh.tell()
    return processed, skipped, new_sites, updated_sites, alerts, new_offset


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest downlink packets into the ground-state SQLite DB.")
    parser.add_argument("--downlink", required=True, help="Path to the JSONL downlink packet file.")
    parser.add_argument("--db", default="ground.db", help="SQLite database path (default: ground.db).")
    parser.add_argument("--follow", action="store_true", help="Continuously ingest newly appended packets.")
    parser.add_argument("--interval-seconds", type=int, default=2, help="Polling interval for --follow mode.")
    args = parser.parse_args()

    downlink_path = Path(args.downlink)
    if not downlink_path.is_file():
        raise SystemExit(f"Downlink file not found: {downlink_path}")

    conn = init_ground_db(Path(args.db))
    offset = 0
    totals = {"processed": 0, "skipped": 0, "new_sites": 0, "updated_sites": 0, "alerts": 0}

    while True:
        processed, skipped, new_sites, updated_sites, alerts, offset = _ingest_available(
            conn,
            downlink_path=downlink_path,
            offset=offset,
        )
        totals["processed"] += processed
        totals["skipped"] += skipped
        totals["new_sites"] += new_sites
        totals["updated_sites"] += updated_sites
        totals["alerts"] += alerts
        print(
            f"Done: processed={totals['processed']} skipped={totals['skipped']}"
            f" new_sites={totals['new_sites']} updated_sites={totals['updated_sites']}"
            f" alerts={totals['alerts']}"
        )
        if not args.follow:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

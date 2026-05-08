from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datacenter_watch.ground_ops import ingest_packet, init_ground_db
from datacenter_watch.satellite_ops import clear_cache, observe_tile


def _payload(
    *,
    bbox: list[float] | None = None,
    site_class: str = "industrial_site",
    construction_stage: str = "active_construction",
    reasoning: str = "ignored",
    image_quality_limited: bool = False,
) -> dict[str, object]:
    return {
        "detections": [
            {
                "bbox": bbox or [0.2, 0.2, 0.4, 0.4],
                "site_class": site_class,
                "construction_stage": construction_stage,
                "roof_bright_membrane": False,
                "bare_soil_present": True,
                "reasoning": reasoning,
            }
        ],
        "tile_context": {
            "image_quality_limited": image_quality_limited,
        },
    }


class OpsPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_reasoning_only_change_is_suppressed(self) -> None:
        first = observe_tile(
            tile_id="dc_1/s00",
            tile_lon=-74.2,
            tile_lat=40.7,
            size_km=5.0,
            observed_at="2026-05-01T00:00:00+00:00",
            payload=_payload(reasoning="first"),
        )
        second = observe_tile(
            tile_id="dc_1/s00",
            tile_lon=-74.2,
            tile_lat=40.7,
            size_km=5.0,
            observed_at="2026-05-02T00:00:00+00:00",
            payload=_payload(reasoning="second"),
        )
        self.assertTrue(first["transmitted"])
        self.assertFalse(second["transmitted"])
        self.assertEqual(second["reason"], "hash_match")

    def test_changed_stage_transmits_packet(self) -> None:
        observe_tile(
            tile_id="dc_1/s00",
            tile_lon=-74.2,
            tile_lat=40.7,
            size_km=5.0,
            observed_at="2026-05-01T00:00:00+00:00",
            payload=_payload(construction_stage="active_construction"),
        )
        second = observe_tile(
            tile_id="dc_1/s00",
            tile_lon=-74.2,
            tile_lat=40.7,
            size_km=5.0,
            observed_at="2026-05-10T00:00:00+00:00",
            payload=_payload(construction_stage="operational"),
        )
        self.assertTrue(second["transmitted"])
        packet = second["packet"]
        self.assertEqual(packet["change_type"], "updated")
        diff = packet["diff"]
        self.assertEqual(diff["detections"][0]["fields"], ["construction_stage"])

    def test_ground_ingest_creates_site_and_queues_high_severity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = init_ground_db(Path(tmpdir) / "ground.db")
            first = observe_tile(
                tile_id="dc_1/s00",
                tile_lon=-74.2,
                tile_lat=40.7,
                size_km=5.0,
                observed_at="2026-05-01T00:00:00+00:00",
                payload=_payload(construction_stage="active_construction"),
            )
            second = observe_tile(
                tile_id="dc_1/s00",
                tile_lon=-74.2,
                tile_lat=40.7,
                size_km=5.0,
                observed_at="2026-05-20T00:00:00+00:00",
                payload=_payload(construction_stage="operational"),
            )
            self.assertTrue(first["transmitted"])
            self.assertTrue(second["transmitted"])

            first_result = ingest_packet(conn, first["packet"])
            second_result = ingest_packet(conn, second["packet"])

            self.assertTrue(first_result["processed"])
            self.assertEqual(first_result["new_sites"], 1)
            self.assertTrue(second_result["processed"])
            self.assertEqual(second_result["updated_sites"], 1)

            site_count = conn.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
            stage_count = conn.execute("SELECT COUNT(*) FROM stage_history").fetchone()[0]
            rapid_count = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE alert_type = 'rapid_construction'"
            ).fetchone()[0]
            queue_count = conn.execute("SELECT COUNT(*) FROM enrichment_queue").fetchone()[0]

            self.assertEqual(site_count, 1)
            self.assertEqual(stage_count, 2)
            self.assertEqual(rapid_count, 1)
            self.assertGreaterEqual(queue_count, 1)

    def test_transmitted_packet_images_are_saved_on_ground(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ground.db"
            conn = init_ground_db(db_path)
            packet_result = observe_tile(
                tile_id="dc_1/s00",
                tile_lon=-74.2,
                tile_lat=40.7,
                size_km=5.0,
                observed_at="2026-05-01T00:00:00+00:00",
                payload=_payload(),
                image_bytes={
                    "rgb": b"rgb-bytes",
                    "swir": b"swir-bytes",
                    "index": b"index-bytes",
                    "mapbox": b"mapbox-bytes",
                },
            )
            self.assertTrue(packet_result["transmitted"])

            ingest_result = ingest_packet(conn, packet_result["packet"])
            self.assertTrue(ingest_result["processed"])

            row = conn.execute(
                "SELECT rgb_path, swir_path, index_path, mapbox_path, packet_json FROM observations"
            ).fetchone()
            self.assertIsNotNone(row)
            rgb_path = Path(row["rgb_path"])
            swir_path = Path(row["swir_path"])
            index_path = Path(row["index_path"])
            mapbox_path = Path(row["mapbox_path"])
            self.assertTrue(rgb_path.exists())
            self.assertTrue(swir_path.exists())
            self.assertTrue(index_path.exists())
            self.assertTrue(mapbox_path.exists())
            self.assertEqual(rgb_path.read_bytes(), b"rgb-bytes")
            self.assertEqual(swir_path.read_bytes(), b"swir-bytes")
            self.assertEqual(index_path.read_bytes(), b"index-bytes")
            self.assertEqual(mapbox_path.read_bytes(), b"mapbox-bytes")
            self.assertNotIn('"images"', str(row["packet_json"]))

    def test_cleared_packet_marks_site_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = init_ground_db(Path(tmpdir) / "ground.db")
            first = observe_tile(
                tile_id="dc_1/s00",
                tile_lon=-74.2,
                tile_lat=40.7,
                size_km=5.0,
                observed_at="2026-05-01T00:00:00+00:00",
                payload=_payload(),
            )
            cleared = observe_tile(
                tile_id="dc_1/s00",
                tile_lon=-74.2,
                tile_lat=40.7,
                size_km=5.0,
                observed_at="2026-05-05T00:00:00+00:00",
                payload={"detections": [], "tile_context": _payload()["tile_context"]},
            )
            ingest_packet(conn, first["packet"])
            cleared_result = ingest_packet(conn, cleared["packet"])

            self.assertTrue(cleared_result["processed"])
            active = conn.execute("SELECT active FROM sites").fetchone()[0]
            cleared_alerts = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE alert_type = 'site_cleared'"
            ).fetchone()[0]
            self.assertEqual(active, 0)
            self.assertEqual(cleared_alerts, 1)


if __name__ == "__main__":
    unittest.main()

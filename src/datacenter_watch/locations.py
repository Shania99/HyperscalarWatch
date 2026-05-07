import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

POSITIVES_CSV = Path("/Users/shaniamitra/Downloads/positives_1000.csv")


@dataclass(frozen=True)
class Location:
    id: str
    lon: float
    lat: float
    expected_category: str  # Dataset category label, not passed to the model


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "location"


def _load_locations(csv_path: Path) -> list[Location]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    base_counts = Counter(_slugify(str(row["name"])) for row in rows)
    seen: defaultdict[str, int] = defaultdict(int)
    locations: list[Location] = []

    for row in rows:
        base_id = _slugify(str(row["name"]))
        seen[base_id] += 1
        loc_id = base_id if base_counts[base_id] == 1 else f"{base_id}_{seen[base_id]}"
        locations.append(
            Location(
                id=loc_id,
                lon=float(str(row["lon"])),
                lat=float(str(row["lat"])),
                expected_category="data_center",
            )
        )

    return locations


LOCATIONS: list[Location] = _load_locations(POSITIVES_CSV)

LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}

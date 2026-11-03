"""SEN-FR-06: check that train/val/test.parquet match their data card.

Run from the repo root:
    python -m sentinel_data.check_processed [folder]

Use it after downloading the processed files from the shared Drive folder
(or after running split) to make sure you have exactly the files the data
card describes. The folder defaults to data/processed/.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from sentinel_data.get_data import sha256

DATA_DIR = Path("data/processed")


def file_info(path):
    """SHA-256 and row count of one Parquet file, as stored in the data card."""
    return {
        "sha256": sha256(path),
        "rows": pq.ParquetFile(path).metadata.num_rows,
    }


def check_processed(data_dir=DATA_DIR):
    """Compare every file listed in data_card.json with what's on disk.

    Returns a list of problems; an empty list means everything matches.
    """
    data_dir = Path(data_dir)
    card = json.loads((data_dir / "data_card.json").read_text())
    problems = []
    for name, expected in card.get("files", {}).items():
        path = data_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        actual = file_info(path)
        if actual["sha256"] != expected["sha256"]:
            problems.append(f"{name}: SHA-256 doesn't match the data card")
        if actual["rows"] != expected["rows"]:
            problems.append(f"{name}: {actual['rows']} rows, data card says {expected['rows']}")
    if not card.get("files"):
        problems.append("data_card.json has no 'files' section (made before SEN-FR-06?)")
    return problems


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    card = json.loads((folder / "data_card.json").read_text())
    problems = check_processed(folder)
    if problems:
        print("ERROR:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    print(f"OK  {card['version']} ({card['date']}): "
          + ", ".join(f"{n} {i['rows']:,} rows" for n, i in card["files"].items()))

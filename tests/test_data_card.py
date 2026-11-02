"""SEN-FR-06: processed data is saved as Parquet with a data card that can be checked."""
import json

import numpy as np
import pandas as pd

from sentinel_data import split as split_mod
from sentinel_data.check_processed import check_processed


def make_clean_parquet(folder, seed=0):
    """Write a small clean.parquet with two benign days and one attack."""
    rng = np.random.default_rng(seed)
    frames = []
    for day, label, n in [("Monday", "BENIGN", 600), ("Tuesday", "BENIGN", 500),
                          ("Tuesday", "SSH-Patator", 200)]:
        frames.append(pd.DataFrame({
            "day": day,
            "label": label,
            "bidirectional_first_seen_ms": rng.integers(0, 3 * 3600 * 1000, n),
            "bidirectional_duration_ms": rng.integers(0, 5_000, n),
            "bidirectional_packets": rng.integers(1, 50, n),
        }))
    pd.concat(frames, ignore_index=True).to_parquet(folder / "clean.parquet", index=False)


def run_split(tmp_path):
    make_clean_parquet(tmp_path)
    split_mod.main(data_dir=tmp_path)
    return json.loads((tmp_path / "data_card.json").read_text())


def test_parquet_files_and_card_written(tmp_path):
    """train/val/test.parquet and data_card.json are all saved."""
    run_split(tmp_path)
    for name in ("train.parquet", "val.parquet", "test.parquet", "data_card.json"):
        assert (tmp_path / name).exists()


def test_card_has_version_date_and_class_counts(tmp_path):
    """The card has what SEN-FR-06 asks for: version, date, rows per class."""
    card = run_split(tmp_path)
    assert card["version"] == split_mod.DATA_VERSION
    assert card["date"]
    assert set(card["rows_per_class"]) == {"BENIGN", "SSH-Patator"}
    for counts in card["rows_per_class"].values():
        assert set(counts) == {"train", "val", "test"}


def test_card_rows_match_files(tmp_path):
    """Row counts in the card match what's actually in each Parquet file."""
    card = run_split(tmp_path)
    for s in ("train", "val", "test"):
        rows = len(pd.read_parquet(tmp_path / f"{s}.parquet"))
        assert card["files"][f"{s}.parquet"]["rows"] == rows == card["rows"][s]


def test_check_passes_on_fresh_files(tmp_path):
    run_split(tmp_path)
    assert check_processed(tmp_path) == []


def test_check_catches_changed_file(tmp_path):
    """If a downloaded file isn't the one in the card, the check says which."""
    run_split(tmp_path)
    df = pd.read_parquet(tmp_path / "val.parquet")
    df.iloc[:-1].to_parquet(tmp_path / "val.parquet", index=False)
    problems = check_processed(tmp_path)
    assert any("val.parquet" in p for p in problems)


def test_check_catches_missing_file(tmp_path):
    run_split(tmp_path)
    (tmp_path / "test.parquet").unlink()
    assert check_processed(tmp_path) == ["test.parquet: missing"]

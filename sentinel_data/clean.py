"""SEN-FR-03: clean the raw NFS-2023-TE tables so models never see broken values.

Run from the repo root (after get_data):
    python -m sentinel_data.clean

Reads data/raw/NFS-2023-TE/<Day>.csv for all five days and writes
data/processed/clean.parquet plus clean_summary.json. The cleaning rules and
the results on the full dataset are in sentinel_data/README.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/NFS-2023-TE")
OUT_DIR = Path("data/processed")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

LABEL = "label"
TIME_COL = "bidirectional_first_seen_ms"

# Kept in the output so the split can order flows by time, but they are
# not model features
META_COLS = ["day", TIME_COL]

# Columns that identify a host, a connection or a point in time instead of
# describing how the traffic behaves. A model trained on these can learn
# "this IP attacked at 3pm" and fall apart on new traffic.
# src_port is a random port picked per connection, so it can also tie flows
# from the same session together across train and test.
ID_COLS = [
    "id", "expiration_id",
    "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui",
    "src_port",
    "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
]

DURATION_COLS = [
    "bidirectional_duration_ms", "src2dst_duration_ms", "dst2src_duration_ms",
]


def load_raw(raw_dir=RAW_DIR, days=DAYS):
    """Load every day's CSV into one table, with a 'day' column added."""
    frames = []
    for day in days:
        df = pd.read_csv(Path(raw_dir) / f"{day}.csv", low_memory=False)
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df):
    """Apply the cleaning rules in order.

    Returns (cleaned table, summary dict with how much each rule removed).
    The input table is left unchanged.
    """
    summary = {"rows_in": len(df)}
    df = df.copy()

    # 1. Turn inf/-inf into NaN, then drop any row with a NaN or no label
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    bad = df[num_cols].isna().any(axis=1) | df[LABEL].isna()
    summary["dropped_nan_or_inf"] = int(bad.sum())
    df = df[~bad]

    # 2. Drop rows with a negative duration
    dur = [c for c in DURATION_COLS if c in df.columns]
    neg = (df[dur] < 0).any(axis=1)
    summary["dropped_negative_duration"] = int(neg.sum())
    df = df[~neg]

    # 3. Drop exact duplicate rows. id is ignored because it's just the row
    # number inside each day's file, so it's different on every row
    dup = df.duplicated(subset=[c for c in df.columns if c != "id"])
    summary["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    # 4. Drop identifier columns
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    summary["dropped_id_columns"] = list(ID_COLS)

    # 5. Drop columns that have the same value in every row
    features = feature_columns(df)
    constant = [c for c in features if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    summary["dropped_constant_columns"] = constant

    summary["rows_out"] = len(df)
    summary["features"] = feature_columns(df)
    return df.reset_index(drop=True), summary


def feature_columns(df):
    """List the model feature columns (everything except label, day and time)."""
    return [c for c in df.columns if c not in META_COLS + [LABEL]]


def print_summary(s):
    """Print how many rows and columns each cleaning rule removed."""
    print(f"Rows in:                   {s['rows_in']:,}")
    print(f"Dropped NaN/inf:           {s['dropped_nan_or_inf']:,}")
    print(f"Dropped negative duration: {s['dropped_negative_duration']:,}")
    print(f"Dropped exact duplicates:  {s['dropped_duplicates']:,}")
    print(f"Rows out:                  {s['rows_out']:,}")
    print(f"Constant columns dropped:  {', '.join(s['dropped_constant_columns']) or 'none'}")
    print(f"Feature columns:           {len(s['features'])}")


def main(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    df, summary = clean(load_raw(raw_dir))
    print_summary(summary)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "clean.parquet", index=False)
    (out_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'clean.parquet'}")


if __name__ == "__main__":
    main()

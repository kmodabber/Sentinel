"""SEN-FR-04: split the cleaned data 60/20/20 by time inside each attack window.

Run from the repo root (after clean):
    python -m sentinel_data.split [--gap-seconds 1]

Reads data/processed/clean.parquet and writes train.parquet, val.parquet,
test.parquet and data_card.json to data/processed/.

A "window" is every flow with the same day and label, e.g. Wednesday's
DoS Hulk, or Monday's benign traffic. Inside each window flows are ordered by
start time, and the points where 60% and 80% of flows have started give two
boundary times, t1 and t2:

    train: flow ENDS before t1 - gap
    val:   flow starts at or after t1 and ENDS before t2 - gap
    test:  flow starts at or after t2

Using the end time means no connection is ever running on both sides of a
boundary. Flows that would cross one are dropped.

Why not a fixed gap on start times: in dense attacks (DoS GoldenEye,
PortScan) the middle 20% of flows can all start within two minutes, so a
120 s gap left them with no validation data at all.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_data.clean import LABEL, TIME_COL, feature_columns

DATA_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")
FRACTIONS = (0.6, 0.2, 0.2)
DURATION_COL = "bidirectional_duration_ms"
GAP_SECONDS = 1
DATA_VERSION = "nfs2023te-v1"

# SEN-FR-05: attack classes with fewer flows than this (in the cleaned data)
# are merged into one "Rare attack" class. With a handful of flows a class
# gets only 1-3 test rows, so its per-class scores would mean nothing.
RARE_MIN_FLOWS = 100
RARE_LABEL = "Rare attack"
ORIGINAL_LABEL = "label_original"
BENIGN = "BENIGN"


def find_rare_classes(df, min_flows=RARE_MIN_FLOWS):
    """Attack labels with fewer than min_flows rows. BENIGN is never rare."""
    counts = df[LABEL].value_counts()
    rare = counts[(counts < min_flows) & (counts.index != BENIGN)]
    return sorted(rare.index)


def merge_rare_classes(df, rare):
    """Relabel the rare classes as RARE_LABEL.

    The old label is kept in label_original so nothing is lost and the
    evaluation can still look at each rare attack on its own.
    """
    out = df.copy()
    out[ORIGINAL_LABEL] = out[LABEL]
    out.loc[out[LABEL].isin(rare), LABEL] = RARE_LABEL
    return out


def assign_window(starts, ends, gap_ms, fractions=FRACTIONS):
    """Label each flow in one window as 'train', 'val', 'test' or 'gap'.

    starts and ends are the flow start and end times in ms. Flows labelled
    'gap' cross a boundary and are dropped later.
    """
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    n = len(starts)
    out = np.full(n, "gap", dtype=object)
    if n == 0:
        return out

    # boundary times: start time of the flow at the 60% and 80% positions
    ordered = np.sort(starts, kind="mergesort")
    t1 = ordered[min(int(n * fractions[0]), n - 1)]
    t2 = ordered[min(int(n * (fractions[0] + fractions[1])), n - 1)]

    out[ends < t1 - gap_ms] = "train"
    out[(starts >= t1) & (ends < t2 - gap_ms)] = "val"
    out[starts >= t2] = "test"
    return out


def split(df, gap_seconds=GAP_SECONDS):
    """Return df with a 'split' column added and boundary-crossing flows removed.

    The same input always gives the same output, whatever order the rows
    are in.
    """
    gap_ms = gap_seconds * 1000
    starts = df[TIME_COL]
    ends = starts + df[DURATION_COL]
    parts = pd.Series("gap", index=df.index, dtype=object)
    for _, idx in df.groupby(["day", LABEL], sort=True).groups.items():
        parts.loc[idx] = assign_window(
            starts.loc[idx].to_numpy(), ends.loc[idx].to_numpy(), gap_ms
        )

    out = df.assign(split=parts)
    out = out[out["split"] != "gap"]
    # sort by index first so flows with the same start time always come out
    # in the same order, whatever order they went in
    out = out.sort_index(kind="mergesort")
    return out.sort_values(["day", TIME_COL, LABEL], kind="mergesort")


def window_counts(df):
    """Table of how many flows each (day, label) window has in train/val/test."""
    counts = (
        df.groupby(["day", LABEL, "split"]).size()
        .unstack("split", fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
    )
    return counts


def data_card(split_df, rows_before, gap_seconds):
    """Summary of the split saved next to the Parquet files: version, date,
    source, rows per class in each part, and the feature list."""
    per_class = (
        split_df.groupby([LABEL, "split"]).size()
        .unstack("split", fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
    )
    return {
        "version": DATA_VERSION,
        "date": date.today().isoformat(),
        "source": "NFS-2023-TE, FlowFrontiers/CyberML-DataQuality @ c6e0191",
        "split": "time-based 60/20/20 inside each (day, label) window; "
                 "a flow must end before the next part starts (plus gap)",
        "gap_seconds": gap_seconds,
        "rows_before_split": rows_before,
        "rows_dropped_in_gaps": rows_before - len(split_df),
        "rows": {s: int((split_df["split"] == s).sum()) for s in SPLITS},
        "rows_per_class": {
            label: {s: int(row[s]) for s in SPLITS}
            for label, row in per_class.iterrows()
        },
        "features": feature_columns(split_df.drop(columns="split")),
    }


def main(data_dir=DATA_DIR, gap_seconds=GAP_SECONDS, rare_min_flows=RARE_MIN_FLOWS):
    data_dir = Path(data_dir)
    df = pd.read_parquet(data_dir / "clean.parquet")
    # windows use the original labels so each rare attack keeps its own
    # time window, then the rare ones are merged after splitting
    out = split(df, gap_seconds)
    rare = find_rare_classes(df, rare_min_flows)

    counts = window_counts(out)
    print(counts.to_string())
    # warn if any window ended up with nothing in one of its parts
    empty = counts[(counts == 0).any(axis=1)]
    if len(empty):
        print("\nWARNING: these windows have an empty part (see SEN-FR-05):")
        print(empty.to_string())

    out = merge_rare_classes(out, rare)
    print(f"\nMerged into '{RARE_LABEL}' (< {rare_min_flows} flows): {', '.join(rare) or 'none'}")

    for s in SPLITS:
        part = out[out["split"] == s].drop(columns="split")
        part.to_parquet(data_dir / f"{s}.parquet", index=False)

    card = data_card(out, len(df), gap_seconds)
    (data_dir / "data_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nRows: {card['rows']}  dropped in gaps: {card['rows_dropped_in_gaps']:,}")
    print(f"Saved train/val/test.parquet and data_card.json to {data_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gap-seconds", type=int, default=GAP_SECONDS)
    p.add_argument("--rare-min-flows", type=int, default=RARE_MIN_FLOWS)
    args = p.parse_args()
    main(gap_seconds=args.gap_seconds, rare_min_flows=args.rare_min_flows)

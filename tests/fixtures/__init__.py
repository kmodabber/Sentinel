"""SEN-109 test fixtures.

Small in-memory DataFrames that mimic data/processed/ layout for baseline tests,
so tests never depend on the real Parquet (which is multi-GB and not committed).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_features.features import FEATURES

# All 15 raw CICIDS labels as they appear in train.parquet before SEN-FR-05 merge.
RAW_CICIDS_LABELS = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]

# After SEN-FR-05 rare-class merge (Heartbleed, Infiltration, Sql Injection, XSS
# each have <100 flows in the real train split): the canonical 12 classes.
CANONICAL_12 = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "Rare attack",
    "SSH-Patator",
    "Web Attack - Brute Force",
]


def make_cicids_frame(label_counts: dict[str, int], seed: int = 0) -> pd.DataFrame:
    """Build a tiny DataFrame shaped like data/processed/*.parquet.

    Produces every column prep_xy / ID_COLS expects, with feature values drawn
    from seeded normals so the arithmetic is deterministic but not identical
    across classes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    row_i = 0
    for label, n in label_counts.items():
        shift = (hash(label) % 7) + 1
        for _ in range(n):
            row = {col: float(rng.normal(shift, 1.0)) for col in FEATURES}
            row["dst_port"] = int(rng.integers(1, 65535))
            row["protocol"] = int(rng.choice([6, 17]))
            row["label"] = label
            row["src_ip"] = "10.0.0.1"
            row["dst_ip"] = "10.0.0.2"
            row["src_mac"] = "aa:bb:cc:dd:ee:ff"
            row["dst_mac"] = "ff:ee:dd:cc:bb:aa"
            row["src_oui"] = "aa:bb:cc"
            row["dst_oui"] = "ff:ee:dd"
            row["flow_id"] = f"flow-{row_i}"
            row["id"] = row_i
            row["expiration_id"] = 0
            row["timestamp"] = row_i
            for col in [
                "bidirectional_first_seen_ms",
                "bidirectional_last_seen_ms",
                "src2dst_first_seen_ms",
                "src2dst_last_seen_ms",
                "dst2src_first_seen_ms",
                "dst2src_last_seen_ms",
            ]:
                row[col] = row_i
            rows.append(row)
            row_i += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_splits(out_dir: Path, label_counts: dict[str, int], seed: int = 0) -> None:
    """Write train/val/test parquet with the same label distribution to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = make_cicids_frame(label_counts, seed=seed + hash(split) % 100)
        df.to_parquet(out_dir / f"{split}.parquet", index=False)

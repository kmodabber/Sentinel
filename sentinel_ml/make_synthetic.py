"""Synthetic network flow generator for local testing before real Parquet splits exist."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_parquet_splits(out_dir=Path("data/processed"), n=6000, seed=0):
    rng = np.random.default_rng(seed)
    # Skew class balance heavily toward BENIGN (~73%) to mirror realistic traffic
    classes = ["BENIGN"] * 8 + ["PortScan", "DoS", "SSH-Patator"]
    labels = rng.choice(classes, size=n)
    shift = pd.Series(labels).map({"BENIGN": 0, "PortScan": 1, "DoS": 2, "SSH-Patator": 3}).to_numpy()

    # Synthesize flow features with class-dependent distribution shifts; PortScan randomizes dst_port
    df = pd.DataFrame({
        "src_ip": "10.0.0.1",
        "dst_port": np.where(shift == 1, rng.integers(1, 65535, n), rng.choice([22, 80, 443], n)),
        "bidirectional_packets": rng.poisson(5 + 3 * shift),
        "bidirectional_bytes": rng.normal(500 + 200 * shift, 150),
        "bidirectional_duration_ms": rng.exponential(100 + 50 * shift),
        "src2dst_mean_ps": rng.normal(80 + 10 * shift, 30),
        "label": labels,
    })

    # Write 60/20/20 train/val/test splits
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = int(0.6 * n), int(0.8 * n)
    df.iloc[:a].to_parquet(out_dir / "train.parquet")
    df.iloc[a:b].to_parquet(out_dir / "val.parquet")
    df.iloc[b:].to_parquet(out_dir / "test.parquet")
    print(f"Saved synthetic train/val/test Parquet files to {out_dir}")


if __name__ == "__main__":
    generate_parquet_splits()

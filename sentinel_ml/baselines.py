#FR07
"""SEN-FR-07: sanity baselines (majority class + logistic regression).

Purpose: show what "too easy" looks like. If these baselines score ~95%+,
the task or the split is leaking (see 08 Test Plan, leakage alarm / risk R2).

Usage
  # Stand-in data, works today before the sample Parquet exists:
  python baselines.py --synthetic

  # Real data (train/val/test Parquet from SEN-FR-06):
  python baselines.py --data-dir data/processed --eval-split val
  python baselines.py --data-dir data/processed --eval-split test   # report only, run once

Both tasks (binary, multi-class) are run, each with and without destination port.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Identifiers that must never be model features (SEN-FR-03, 08 Test Plan data checks).
# NFStream names; adjust once the real column list is confirmed against features.json.
ID_COLS = {
    "id", "expiration_id", "flow_id",
    "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "timestamp",
}
DST_PORT = "dst_port"
LEAKAGE_ALARM = 0.999


# ---------------------------------------------------------------- data
def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in ("train", "val", "test")}


def synthetic_splits(n: int = 6000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Fake flows in roughly the real shape, so the script can be built before data exists."""
    rng = np.random.default_rng(seed)
    classes = ["BENIGN"] * 8 + ["PortScan", "DoS", "SSH-Patator"]
    label = rng.choice(classes, size=n)
    shift = pd.Series(label).map({"BENIGN": 0, "PortScan": 1, "DoS": 2, "SSH-Patator": 3}).to_numpy()
    df = pd.DataFrame({
        "src_ip": [f"10.0.0.{i % 250}" for i in range(n)],
        "dst_ip": "192.168.1.5",
        "bidirectional_first_seen_ms": np.arange(n) * 10,
        "dst_port": np.where(shift == 1, rng.integers(1, 65535, n), rng.choice([22, 80, 443], n)),
        "bidirectional_packets": rng.poisson(5 + 3 * shift),
        "bidirectional_bytes": rng.normal(500 + 200 * shift, 150),
        "bidirectional_duration_ms": rng.exponential(100 + 50 * shift),
        "src2dst_mean_ps": rng.normal(80 + 10 * shift, 30),
        "label": label,
    })
    a, b = int(0.6 * n), int(0.8 * n)  # time-ordered 60/20/20, like the real split
    return {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:]}


def feature_matrix(df: pd.DataFrame, label_col: str, use_dst_port: bool) -> pd.DataFrame:
    drop = ID_COLS | {label_col} | (set() if use_dst_port else {DST_PORT})
    X = df.drop(columns=[c for c in df.columns if c in drop])
    X = X.select_dtypes(include="number")  # strings like application_name are not baseline features
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in features: the cleaning step (SEN-FR-03) should have removed these")
    return X


def targets(df: pd.DataFrame, label_col: str, benign: str, task: str) -> pd.Series:
    y = df[label_col].astype(str)
    return (y != benign).map({False: "benign", True: "attack"}) if task == "binary" else y


# ---------------------------------------------------------------- models
def make_models(seed: int) -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(
            StandardScaler(),  # flow features span many orders of magnitude
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        ),
    }


def evaluate(model, X, y, benign_name: str) -> dict:
    pred = model.predict(X)
    labels = sorted(y.unique())
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    benign_mask = (y == benign_name).to_numpy()
    fpr = float((pred[benign_mask] != benign_name).mean()) if benign_mask.any() else None
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "benign_fpr": fpr,
        "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                      for c in labels},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }


# ---------------------------------------------------------------- main
def run(splits, label_col, benign, eval_split, seed) -> list[dict]:
    results = []
    for task in ("binary", "multiclass"):
        benign_name = "benign" if task == "binary" else benign
        y_tr = targets(splits["train"], label_col, benign, task)
        y_ev = targets(splits[eval_split], label_col, benign, task)
        for use_port in (True, False):
            X_tr = feature_matrix(splits["train"], label_col, use_port)
            X_ev = feature_matrix(splits[eval_split], label_col, use_port)
            for name, model in make_models(seed).items():
                model.fit(X_tr, y_tr)
                r = evaluate(model, X_ev, y_ev, benign_name)
                r.update(model=name, task=task, dst_port=use_port, eval_split=eval_split,
                         n_features=X_tr.shape[1])
                results.append(r)
                flag = "  <-- LEAKAGE ALARM, investigate" if r["macro_f1"] >= LEAKAGE_ALARM else ""
                fpr = f"{r['benign_fpr']:.4f}" if r["benign_fpr"] is not None else "n/a"
                print(f"{task:10} {name:8} dst_port={str(use_port):5} "
                      f"macro-F1={r['macro_f1']:.4f} benign-FPR={fpr}{flag}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--synthetic", action="store_true", help="use fake stand-in data")
    p.add_argument("--label-col", default="label")
    p.add_argument("--benign", default="BENIGN", help="name of the benign class in the label column")
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--wandb", action="store_true", help="log to the team W&B project (SEN-FR-10)")
    args = p.parse_args()

    if not args.synthetic and not args.data_dir:
        p.error("give --data-dir or --synthetic")
    splits = synthetic_splits() if args.synthetic else load_splits(args.data_dir)
    results = run(splits, args.label_col, args.benign, args.eval_split, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"baselines_{args.eval_split}{'_synthetic' if args.synthetic else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")

    if args.wandb:
        import wandb
        for r in results:
            with wandb.init(project="sentinel", job_type="baseline", config={
                k: r[k] for k in ("model", "task", "dst_port", "eval_split", "n_features")}, reinit=True) as run_:
                run_.log({"macro_f1": r["macro_f1"], "benign_fpr": r["benign_fpr"]})


if __name__ == "__main__":
    main()


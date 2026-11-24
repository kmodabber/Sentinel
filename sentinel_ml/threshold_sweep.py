"""SEN-134: pick the sensor's default alert confidence threshold on VAL only.

Scores validation flows through the exact sensor path
(sensor.package.load_package + sensor.package.score), then sweeps the alert
threshold from 0.50 to 0.99 and reports, at each step:

- binary attack recall (attack flows that become alerts)
- benign false-positive rate (benign flows that become alerts)
- alerts per 10,000 benign flows
- per-class recall (for every attack class)

Writes a per-class thresholds.json in the C1 shape (just the numbers; the
package owner copies it into the real package folder when they publish it).
Also writes a per-threshold CSV and a PNG of recall/FPR vs threshold so the
write-up can reference them.

Everything is val-only (SEN-NFR-06): the test set is for reporting, never for
picking anything.
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# sensor.package imports nfstream, whose native extension won't load on bare
# Windows; keep that import inside score_val() so unit tests (which only
# exercise the metric functions) don't need a working nfstream.
from sentinel_features.features import ATTACK_LABELS, BENIGN

RESULTS_DIR = Path("results") / "sen-134"
DATA_DIR = Path("data") / "processed"
DEFAULT_SWEEP = np.round(np.arange(0.50, 1.00, 0.01), 2)
# SEN-NFR-04 target: attack recall >= 0.95 and benign FPR <= 0.01 on val.
RECALL_TARGET = 0.95
FPR_TARGET = 0.01


def _stub_nfstream_on_windows() -> None:
    """Dev-only: let `import nfstream` succeed on bare Windows.

    The sensor's load_package() calls check_nfstream_version() which reads
    nfstream.__version__. On WSL/Linux the real nfstream loads fine and the
    check is a real check. On Windows the native extension won't load and
    sensor.package can't even import. We only use score() here (not the
    live capture pipeline), so stubbing nfstream to the pinned version is
    safe for the threshold sweep; the sensor itself still enforces the
    real guard at runtime.
    """
    import sys
    if "nfstream" in sys.modules:
        return
    try:
        import nfstream  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("nfstream")
        stub.__version__ = "6.6.0"
        sys.modules["nfstream"] = stub


def score_val(
    package_path: Path,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Score val.parquet through sensor.package.score() exactly as the sensor does.

    Returns a frame with columns: predicted_class, confidence, model_version, label.
    """
    _stub_nfstream_on_windows()
    from sensor.package import load_package, score  # lazy: nfstream DLL on Windows
    package = load_package(package_path)
    val_df = pd.read_parquet(data_dir / "val.parquet")

    # prep_xy strips ID cols + (optionally) dst_port and keeps numeric features.
    # The sensor passes full-feature frames to score(), so rebuild a 56-column
    # frame in FEATURES order; preprocess.joblib handles the drop internally.
    X = val_df[package["features"]].copy()
    detections = score(package, X)
    detections["label"] = val_df["label"].values
    return detections


def _metrics_at(
    labels: np.ndarray,
    preds: np.ndarray,
    conf: np.ndarray,
    threshold: float,
    attack_labels: list[str],
    benign: str = BENIGN,
) -> dict:
    """Compute recall/FPR/per-class recall at a given threshold.

    Alert rule matches SEN-FR-18: the predicted class must be in
    ATTACK_LABELS AND confidence must be >= threshold.
    """
    is_attack_pred = np.isin(preds, attack_labels)
    is_alert = is_attack_pred & (conf >= threshold)

    is_attack_true = labels != benign
    is_benign_true = ~is_attack_true

    n_attack = int(is_attack_true.sum())
    n_benign = int(is_benign_true.sum())

    attack_recall = float((is_alert & is_attack_true).sum() / n_attack) if n_attack else float("nan")
    benign_fpr = float((is_alert & is_benign_true).sum() / n_benign) if n_benign else float("nan")
    alerts_per_10k_benign = float(10_000 * benign_fpr) if n_benign else float("nan")

    per_class = {}
    for cls in attack_labels:
        mask = labels == cls
        total = int(mask.sum())
        if total == 0:
            per_class[cls] = None
            continue
        hits = int((is_alert & mask).sum())
        per_class[cls] = round(hits / total, 4)

    return {
        "threshold": round(float(threshold), 4),
        "attack_recall": round(attack_recall, 4),
        "benign_fpr": round(benign_fpr, 6),
        "alerts_per_10k_benign": round(alerts_per_10k_benign, 2),
        "per_class_recall": per_class,
    }


def sweep(
    detections: pd.DataFrame,
    thresholds: np.ndarray = DEFAULT_SWEEP,
    attack_labels: list[str] = ATTACK_LABELS,
) -> list[dict]:
    """Run the alert rule at each threshold and return one dict per threshold."""
    labels = detections["label"].to_numpy()
    preds = detections["predicted_class"].to_numpy()
    conf = detections["confidence"].to_numpy()
    rows = [_metrics_at(labels, preds, conf, t, attack_labels) for t in thresholds]

    # Monotonicity sanity check (TC-07): as threshold rises, alerts never go up.
    last_alerts = None
    for row in rows:
        if last_alerts is not None and row["alerts_per_10k_benign"] > last_alerts + 1e-9:
            print(
                f"[threshold_sweep] WARNING: benign alerts not monotonic at "
                f"threshold={row['threshold']:.2f} ({row['alerts_per_10k_benign']} > {last_alerts})"
            )
        last_alerts = row["alerts_per_10k_benign"]

    return rows


def recommend_global(rows: list[dict], recall_target: float = RECALL_TARGET,
                     fpr_target: float = FPR_TARGET) -> dict:
    """Highest threshold that still meets attack recall and benign FPR targets.

    If no threshold meets both targets, return the one closest to them, flagged.
    """
    feasible = [r for r in rows if r["attack_recall"] >= recall_target and r["benign_fpr"] <= fpr_target]
    if feasible:
        best = max(feasible, key=lambda r: r["threshold"])
        return {"threshold": best["threshold"], "meets_targets": True, "basis": best}

    # No threshold met both targets. Pick the one with the smallest sum of
    # shortfalls, so the write-up reports the best achievable trade-off.
    def shortfall(r):
        rs = max(0.0, recall_target - r["attack_recall"])
        fs = max(0.0, r["benign_fpr"] - fpr_target)
        return rs + fs

    best = min(rows, key=shortfall)
    return {"threshold": best["threshold"], "meets_targets": False, "basis": best}


def recommend_per_class(rows: list[dict], global_t: float,
                        recall_floor: float = 0.90) -> dict:
    """Per-class thresholds for classes whose recall at global_t is far below.

    If a class never reaches recall_floor at any threshold, keep its global
    value and flag it in the write-up instead.
    """
    attack_labels = list(rows[0]["per_class_recall"].keys())
    per_class = {}
    # Find the row at the global threshold.
    base_row = next(r for r in rows if r["threshold"] == global_t)
    for cls in attack_labels:
        base_recall = base_row["per_class_recall"].get(cls)
        if base_recall is None or base_recall >= recall_floor:
            per_class[cls] = global_t
            continue
        # Lower the threshold for this class until it meets the floor.
        lowered = None
        for r in sorted(rows, key=lambda r: -r["threshold"]):
            if (r["per_class_recall"].get(cls) or 0.0) >= recall_floor:
                lowered = r["threshold"]
        per_class[cls] = lowered if lowered is not None else global_t
    return per_class


def write_outputs(
    rows: list[dict],
    recommendation: dict,
    per_class_thresholds: dict,
    out_dir: Path = RESULTS_DIR,
) -> None:
    """Write thresholds.json (C1 shape), the full sweep CSV, and a PNG plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # thresholds.json in C1 shape: {attack_label: value}. This is NOT the
    # package folder; the package owner copies this into theirs when
    # publishing. Benign is not in this file (dummy_package.py pattern).
    (out_dir / "thresholds.json").write_text(json.dumps(per_class_thresholds, indent=2))

    # Full per-threshold table (handy to paste into the write-up).
    flat = []
    for r in rows:
        line = {"threshold": r["threshold"], "attack_recall": r["attack_recall"],
                "benign_fpr": r["benign_fpr"], "alerts_per_10k_benign": r["alerts_per_10k_benign"]}
        for cls, val in r["per_class_recall"].items():
            line[f"recall_{cls}"] = val
        flat.append(line)
    pd.DataFrame(flat).to_csv(out_dir / "sweep.csv", index=False)

    (out_dir / "recommendation.json").write_text(json.dumps(recommendation, indent=2))

    # PNG: attack recall and benign FPR against threshold.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r["threshold"] for r in rows]
        recalls = [r["attack_recall"] for r in rows]
        fprs = [r["benign_fpr"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.plot(ts, recalls, "b-", label="attack recall (val)")
        ax1.set_xlabel("confidence threshold")
        ax1.set_ylabel("attack recall", color="b")
        ax1.axhline(RECALL_TARGET, color="b", linestyle=":", alpha=0.4, label=f"target {RECALL_TARGET}")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(ts, fprs, "r-", label="benign FPR (val)")
        ax2.set_ylabel("benign FPR", color="r")
        ax2.axhline(FPR_TARGET, color="r", linestyle=":", alpha=0.4, label=f"target {FPR_TARGET}")
        ax2.tick_params(axis="y", labelcolor="r")

        ax1.axvline(recommendation["threshold"], color="k", linestyle="--", alpha=0.6)
        fig.suptitle(f"SEN-134: recall + FPR vs threshold (recommended = {recommendation['threshold']})")
        fig.tight_layout()
        fig.savefig(out_dir / "sweep.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("[threshold_sweep] matplotlib not installed; skipping PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True,
                        help="Path to a C1 model package folder")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--fpr-target", type=float, default=FPR_TARGET)
    args = parser.parse_args()

    print(f"[threshold_sweep] scoring val through {args.package}...", flush=True)
    detections = score_val(args.package, args.data_dir)
    print(f"[threshold_sweep] scored {len(detections):,} val rows", flush=True)

    rows = sweep(detections)
    rec = recommend_global(rows, args.recall_target, args.fpr_target)
    pcls = recommend_per_class(rows, rec["threshold"])
    write_outputs(rows, rec, pcls, args.out_dir)

    print(f"[threshold_sweep] recommended threshold = {rec['threshold']} "
          f"(meets targets={rec['meets_targets']}); wrote {args.out_dir}")


if __name__ == "__main__":
    main()

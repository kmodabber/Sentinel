"""SEN-69 global-check: mean |SHAP| on a val sample for the leakage guard (R2).

Runs TreeSHAP on a bounded val sample, averages |SHAP| across classes and
rows, and reports the top-10 features. Also times the per-alert cost so we
can compare with SEN-NFR-01 (median flow-end-to-alert <= 10 s).

Writes results/sen-69/global.json and docs/eval/sen-69-shap.md.
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_features.features import ATTACK_LABELS
from sentinel_ml.explain import Explainer
from sentinel_ml.threshold_sweep import _stub_nfstream_on_windows

RESULTS_DIR = Path("results") / "sen-69"
DATA_DIR = Path("data") / "processed"
DEFAULT_SAMPLE = 20_000
DOMINANCE_THRESHOLD = 0.60  # one feature carrying >60% of mean |SHAP| is a R2 flag


def run(package_path: Path, data_dir: Path, sample: int, out_dir: Path,
        threshold: float) -> dict:
    _stub_nfstream_on_windows()
    from sensor.package import load_package, score

    package = load_package(package_path)
    val_df = pd.read_parquet(data_dir / "val.parquet")
    if sample < len(val_df):
        val_df = val_df.sample(n=sample, random_state=42).reset_index(drop=True)

    X = val_df[package["features"]].copy()
    detections = score(package, X)
    predicted = detections["predicted_class"].to_numpy()
    confidence = detections["confidence"].to_numpy()
    is_attack_pred = np.isin(predicted, ATTACK_LABELS)
    alert_mask = is_attack_pred & (confidence >= threshold)
    n_alerts = int(alert_mask.sum())

    # Only alerts need explanations in production; mirror that path.
    alert_X = X[alert_mask]
    alert_preds = predicted[alert_mask]

    explainer = Explainer(package)
    t0 = time.time()
    reasons = explainer.explain(alert_X, alert_preds)
    elapsed = time.time() - t0
    ms_per_1000 = (elapsed / max(n_alerts, 1)) * 1000 * 1000

    # Flatten the top-5 shap entries per row into (feature -> list of |shap|).
    agg = {f: [] for f in explainer.feature_names}
    for r in reasons:
        for entry in r["reasons"]:
            agg[entry["feature"]].append(abs(entry["shap"]))

    means = {f: float(np.mean(v)) if v else 0.0 for f, v in agg.items()}
    total = sum(means.values()) or 1.0
    shares = {f: round(means[f] / total, 4) for f in means}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])

    top10 = [{"feature": f, "mean_abs_shap": round(m, 6), "share": shares[f]}
             for f, m in ranked[:10]]
    top_share = top10[0]["share"] if top10 else 0.0
    dominated = top_share >= DOMINANCE_THRESHOLD

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "package": str(package_path),
        "model_version": package["version"],
        "val_sample_rows": int(len(val_df)),
        "alerts_in_sample": n_alerts,
        "threshold": threshold,
        "explain_ms_per_1000_alerts": round(ms_per_1000, 2),
        "top10": top10,
        "dominance_flag": dominated,
        "dominance_threshold": DOMINANCE_THRESHOLD,
    }
    (out_dir / "global.json").write_text(json.dumps(report, indent=2))

    if dominated:
        print(
            f"[shap_global] R2 WARN: top feature '{top10[0]['feature']}' carries "
            f"{top_share:.2%} of mean |SHAP| (threshold {DOMINANCE_THRESHOLD:.0%}).",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Alert threshold; defaults to 0.80 placeholder, "
                             "pass the SEN-134 value once known.")
    args = parser.parse_args()
    report = run(args.package, args.data_dir, args.sample, args.out_dir, args.threshold)
    print(json.dumps(report["top10"], indent=2))


if __name__ == "__main__":
    main()

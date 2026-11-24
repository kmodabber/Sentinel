"""Local dev stand-in model for SEN-134 / SEN-69 / SEN-135.

Trains the multi-class LightGBM without dst_port on data/processed/train.parquet,
using the val-tuned params from Mikhai's rf-lgbm-ensemble-comparison branch
(eval_report_v1.md row `xwewtupk`: n_estimators=100, num_leaves=63, lr=0.1,
reg_lambda=10.0, max_delta_step=2.0, seed=42).

Writes a contract-C1 package folder (default models/dev-lgbm-noport/) so that
sensor.package.load_package() and sensor.package.score() work on it unchanged.

NOT the team package (SEN-68). Someone else exports the real one; this is only
to unblock SEN-134 / SEN-69 / SEN-135 development on my side. Every later script
takes --package so it can be rerun against the real package later.
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay first; see note below.
# LightGBM must be imported before pandas on this Windows env (numpy 2.0 +
# lightgbm 4.7.0 ABI mismatch: fit() segfaults in C-level set_label if
# pandas is imported first). Keep this first.
import lightgbm  # noqa: F401

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sentinel_features.features import ATTACK_LABELS, BENIGN, FEATURES, LABELS
from sentinel_features.nfstream_settings import SETTINGS
from sentinel_ml.eval_template import prep_xy
from sentinel_ml.wrappers import StringClassLGBM, drop_dst_port


def _default_lgbm_multiclass(random_state: int, n_estimators: int = 100) -> LGBMClassifier:
    """Stabilised multi-class LightGBM — inlined from sentinel_ml.ensemble so this
    script does not depend on rf-lgbm-ensemble-comparison code. Params match
    eval_report_v1.md row `xwewtupk` after LGBM_OVERRIDES are applied below.
    """
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_delta_step=2.0,
        reg_lambda=10.0,
        min_child_weight=1.0,
        subsample_for_bin=200_000,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )

VERSION = "dev-lgbm-noport-v0"
OUT_DIR = Path("models") / "dev-lgbm-noport"
DATA_DIR = Path("data") / "processed"
SEED = 42

# Val-tuned no-port multi-class params from eval_report_v1.md (xwewtupk).
LGBM_OVERRIDES = {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 100}
DEFAULT_THRESHOLD = 0.80  # placeholder; SEN-134 writes the real thresholds.json

MODEL_CARD = """# {version}

**Local dev stand-in for SEN-134 / SEN-69 / SEN-135. Not the team package (SEN-68).**

Someone else exports the real package. This stand-in exists only so threshold
sweeps, SHAP explanations and the demo-slice check can be developed against a
model with the shape the sensor expects.

- Built: {date}
- Data version: {data_version}
- Train rows: {train_rows}
- Model: LightGBM multi-class, no dst_port
- Val-tuned params (eval_report_v1.md row `xwewtupk`):
  `n_estimators=100, num_leaves=63, learning_rate=0.1, reg_lambda=10.0, max_delta_step=2.0, seed=42`
- Preprocessing: drop `dst_port` column before scoring (features.json still
  lists all 56 names so the sensor's check_features() passes).
- Val macro-F1: {val_macro_f1}
- Val binary attack recall: {val_binary_attack_recall}
- Known weaknesses: trained without the squad-wide eval harness; test metrics
  are reported in metrics.json but were NOT used to tune anything.
"""


def build_preprocess() -> Pipeline:
    """A pipeline that drops dst_port when the sensor calls preprocess.transform."""
    return Pipeline(steps=[
        ("drop_dst_port", FunctionTransformer(drop_dst_port, validate=False)),
    ])


def _binary_attack_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of attack flows predicted as any attack class."""
    is_attack = y_true != BENIGN
    if not is_attack.any():
        return float("nan")
    pred_attack = y_pred != BENIGN
    return float((pred_attack & is_attack).sum() / is_attack.sum())


def train_and_save(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    version: str = VERSION,
    seed: int = SEED,
) -> Path:
    """Train the LightGBM stand-in and write a C1-shaped folder to out_dir."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    X_tr, y_tr, _ = prep_xy(train_df, use_port=False, task="multiclass")
    X_va, y_va, _ = prep_xy(val_df, use_port=False, task="multiclass")
    X_te, y_te, _ = prep_xy(test_df, use_port=False, task="multiclass")

    y_tr_arr = np.asarray(y_tr)
    y_va_arr = np.asarray(y_va)
    y_te_arr = np.asarray(y_te)

    expected_cols = [f for f in FEATURES if f != "dst_port"]
    if list(X_tr.columns) != expected_cols:
        missing = [c for c in expected_cols if c not in X_tr.columns]
        extra = [c for c in X_tr.columns if c not in expected_cols]
        raise RuntimeError(
            "prep_xy did not yield the expected 55-feature frame: "
            f"missing={missing}, extra={extra}"
        )

    base = _default_lgbm_multiclass(random_state=seed)
    base.set_params(**LGBM_OVERRIDES)
    model = StringClassLGBM(base)
    print(f"[dev_model] fitting LightGBM on {len(X_tr):,} train rows ({X_tr.shape[1]} features, seed={seed})...", flush=True)
    model.fit(X_tr, y_tr_arr)

    preds_va = model.predict(X_va)
    preds_te = model.predict(X_te)

    from sklearn.metrics import f1_score
    val_macro_f1 = float(f1_score(y_va_arr, preds_va, average="macro", zero_division=0))
    test_macro_f1 = float(f1_score(y_te_arr, preds_te, average="macro", zero_division=0))
    val_bin_recall = _binary_attack_recall(y_va_arr, preds_va)
    test_bin_recall = _binary_attack_recall(y_te_arr, preds_te)

    # Leakage guard (R2): hard stop if scores are implausibly high.
    if test_macro_f1 >= 0.999:
        raise RuntimeError(
            f"test macro-F1 {test_macro_f1:.4f} >= 0.999 — possible leakage (R2). "
            "Refusing to write the package; inspect the data and params first."
        )

    data_card = data_dir / "data_card.json"
    data_version = "nfs2023te-v1"
    if data_card.exists():
        try:
            data_version = json.loads(data_card.read_text()).get("version", data_version)
        except Exception:
            pass

    preprocess = build_preprocess()
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(preprocess, out_dir / "preprocess.joblib")

    files = {
        "features.json": {
            "version": version,
            "features": [{"name": f, "type": "number"} for f in FEATURES],
        },
        "labels.json": {"classes": LABELS, "attack": ATTACK_LABELS},
        "nfstream_settings.json": SETTINGS,
        "thresholds.json": {label: DEFAULT_THRESHOLD for label in ATTACK_LABELS},
        "metrics.json": {
            "note": "Local dev stand-in; not the team package. Test metrics reported, not used for tuning.",
            "data_version": data_version,
            "seed": seed,
            "n_features": int(X_tr.shape[1]),
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_va)),
            "test_rows": int(len(X_te)),
            "val_macro_f1": round(val_macro_f1, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "val_binary_attack_recall": round(val_bin_recall, 4),
            "test_binary_attack_recall": round(test_bin_recall, 4),
            "lgbm_params": LGBM_OVERRIDES,
        },
    }
    for name, content in files.items():
        (out_dir / name).write_text(json.dumps(content, indent=2))

    (out_dir / "MODEL_CARD.md").write_text(
        MODEL_CARD.format(
            version=version,
            date=date.today().isoformat(),
            data_version=data_version,
            train_rows=f"{len(X_tr):,}",
            val_macro_f1=f"{val_macro_f1:.4f}",
            val_binary_attack_recall=f"{val_bin_recall:.4f}",
        )
    )

    print(
        f"[dev_model] wrote {out_dir} | "
        f"val macro-F1={val_macro_f1:.4f} test macro-F1={test_macro_f1:.4f} | "
        f"val attack recall={val_bin_recall:.4f} test attack recall={test_bin_recall:.4f}",
        flush=True,
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_and_save(args.data_dir, args.out_dir, args.version, args.seed)


if __name__ == "__main__":
    main()

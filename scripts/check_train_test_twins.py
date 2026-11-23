"""SEN-106: Check exact feature+label twins between train.parquet and test.parquet
and compare Logistic Regression macro-F1 / attack recall / benign FPR with and
without twins.

Twin definition:
- Starting from train.parquet and test.parquet (12 canonical classes after SEN-FR-05),
  all identifier and timestamp columns in sentinel_ml.eval_template.ID_COLS (plus
  non-numeric metadata like 'day' and 'label_original') are excluded via prep_xy().
- With dst_port=True: 56 numeric flow features + target label y.
- With dst_port=False: 55 numeric flow features (dst_port dropped) + target label y.
- A test row is counted as an exact twin in train if its (numeric features X, target y)
  tuple appears at least once in train.parquet.

Model configuration:
- Matches sentinel_ml.baselines._build_models()["logreg"]:
  make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
  max_iter=2000, random_state=42, solver="lbfgs")) fitted on the full train split.

Run from the repo root:
    python scripts/check_train_test_twins.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel_ml.baselines import SEED, canonicalize_labels
from sentinel_ml.eval_template import evaluate_single, prep_xy

DATA_DIR = Path("data/processed")


def find_twin_mask(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_port: bool,
    task: str = "binary",
) -> tuple[np.ndarray, list[str]]:
    """Return boolean mask over test_df where (features, label) exactly matches a train row."""
    X_tr, y_tr, _ = prep_xy(train_df, use_port=use_port, task=task)
    X_te, y_te, _ = prep_xy(test_df, use_port=use_port, task=task)
    feature_cols = list(X_tr.columns)

    tr_key = X_tr.assign(_label=np.asarray(y_tr, dtype=str))
    te_key = X_te.assign(_label=np.asarray(y_te, dtype=str))

    tr_hashes = pd.util.hash_pandas_object(tr_key, index=False)
    te_hashes = pd.util.hash_pandas_object(te_key, index=False)
    mask = te_hashes.isin(set(tr_hashes.to_numpy())).to_numpy(dtype=bool)
    return mask, feature_cols


def _build_logreg(seed: int = SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def run_twins_check(
    data_dir: Path = DATA_DIR,
    train_sample: int | None = None,
    seed: int = SEED,
    check_canonical: bool = True,
) -> dict:
    """Compute twin counts and evaluate baseline Logistic Regression on all vs. unseen test rows."""
    train_full = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    if check_canonical:
        canonicalize_labels(train_full)
        canonicalize_labels(test_df)

    if train_sample is not None and train_sample < len(train_full):
        train_fit = train_full.sample(n=train_sample, random_state=seed).reset_index(drop=True)
    else:
        train_fit = train_full

    summary: dict = {
        "seed": seed,
        "train_rows_total": int(len(train_full)),
        "train_rows_fit": int(len(train_fit)),
        "test_rows_total": int(len(test_df)),
        "variants": {},
    }

    for use_port in (True, False):
        tag = "with_port" if use_port else "no_port"
        twin_mask_bin, feature_cols = find_twin_mask(
            train_full, test_df, use_port=use_port, task="binary"
        )
        twin_mask_mc, _ = find_twin_mask(
            train_full, test_df, use_port=use_port, task="multiclass"
        )
        n_twins = int(twin_mask_bin.sum())
        n_total = int(len(test_df))
        n_unseen = int((~twin_mask_bin).sum())
        twin_pct = round(100.0 * n_twins / n_total, 2) if n_total else 0.0

        print(
            f"\n=== dst_port={use_port} ({len(feature_cols)} features) ===\n"
            f"Exact twins in train (binary label):     {n_twins:,} / {n_total:,} ({twin_pct:.1f}%) "
            f"-> {n_unseen:,} strictly unseen test rows\n"
            f"Exact twins in train (multiclass label): {int(twin_mask_mc.sum()):,} / {n_total:,}",
            flush=True,
        )

        variant_info: dict = {
            "dst_port": use_port,
            "n_features": len(feature_cols),
            "twin_test_rows": n_twins,
            "twin_test_rows_multiclass": int(twin_mask_mc.sum()),
            "unseen_test_rows": n_unseen,
            "unseen_test_rows_multiclass": int((~twin_mask_mc).sum()),
            "twin_pct": twin_pct,
            "tasks": {},
        }

        for task, twin_mask in (("binary", twin_mask_bin), ("multiclass", twin_mask_mc)):
            unseen_mask = ~twin_mask
            X_tr, y_tr, _ = prep_xy(train_fit, use_port=use_port, task=task)
            X_te, y_te, benign_name = prep_xy(test_df, use_port=use_port, task=task)

            model = _build_logreg(seed=seed).fit(X_tr, y_tr)
            all_metrics = evaluate_single(model, X_te, y_te, benign_name)
            unseen_metrics = evaluate_single(
                model,
                X_te.loc[unseen_mask].reset_index(drop=True),
                np.asarray(y_te)[unseen_mask],
                benign_name,
            )

            delta_f1 = round(unseen_metrics["macro_f1"] - all_metrics["macro_f1"], 4)
            entry = {
                "all_test": {
                    "macro_f1": all_metrics["macro_f1"],
                    "benign_fpr": all_metrics["benign_fpr"],
                    "per_class": all_metrics["per_class"],
                },
                "without_twins": {
                    "macro_f1": unseen_metrics["macro_f1"],
                    "benign_fpr": unseen_metrics["benign_fpr"],
                    "per_class": unseen_metrics["per_class"],
                },
                "delta_macro_f1": delta_f1,
            }
            if task == "binary":
                atk_rec_all = all_metrics["per_class"].get("attack", {}).get("recall")
                atk_rec_unseen = unseen_metrics["per_class"].get("attack", {}).get("recall")
                entry["all_test"]["attack_recall"] = atk_rec_all
                entry["without_twins"]["attack_recall"] = atk_rec_unseen
                print(
                    f"  binary     | Macro-F1: {all_metrics['macro_f1']:.4f} -> "
                    f"{unseen_metrics['macro_f1']:.4f} ({delta_f1:+.4f}) | "
                    f"Attack recall: {atk_rec_all:.4f} -> {atk_rec_unseen:.4f} | "
                    f"Benign FPR: {all_metrics['benign_fpr']:.4f} -> {unseen_metrics['benign_fpr']:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"  multiclass | Macro-F1: {all_metrics['macro_f1']:.4f} -> "
                    f"{unseen_metrics['macro_f1']:.4f} ({delta_f1:+.4f}) | "
                    f"Benign FPR: {all_metrics['benign_fpr']:.4f} -> {unseen_metrics['benign_fpr']:.4f}",
                    flush=True,
                )
            variant_info["tasks"][task] = entry

        summary["variants"][tag] = variant_info

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SEN-106 train-test twins check")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--train-sample",
        type=int,
        default=None,
        help="Optional subsample of train.parquet for fitting (default: full train, matching baselines.py).",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/sen_106_twins.json"),
        help="Path to write JSON summary.",
    )
    args = p.parse_args()

    res = run_twins_check(
        data_dir=args.data_dir,
        train_sample=args.train_sample,
        seed=args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2))
    print(f"\nSaved summary to {args.out_json}")

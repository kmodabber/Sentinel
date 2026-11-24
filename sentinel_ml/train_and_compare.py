"""Train Random Forest and LightGBM for binary and multi-class detection, compare them,
and compare their ensemble using a Meta-Learner (Stacking) vs. Fixed Weights with W&B logging.

Run from the repo root:
    python -m sentinel_ml.train_and_compare
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from sentinel_ml.ensemble import (
    FixedWeightEnsemble,
    MetaLearnerEnsemble,
    _normalize_importances,
    _subsample_for_meta,
    build_meta_features,
    default_lgbm,
    default_rf,
    sweep_fixed_weights,
)
from sentinel_ml.eval_template import (
    SEED,
    _json_safe,
    _model_params,
    extract_model_params,
    find_train_twins,
    get_code_commit,
    get_data_metadata,
    get_wandb_init_kwargs,
    prep_xy,
)

DATA_DIR = Path("data/processed")
OUT_DIR = Path("results")
USE_WANDB = True
WANDB_ENTITY = "wcs-sentinel"
WANDB_PROJECT = "Sentinel"


def _wandb_init_kwargs(job_type: str, name: str, config: dict) -> dict:
    """Builds wandb.init arguments, falling back to offline mode when no API key is configured."""
    return get_wandb_init_kwargs(job_type=job_type, name=name, config=config)


# Compact val-only grids (~9 candidates each). Test is never used here.
RF_VAL_GRID = {
    "max_depth": [20, 26, None],
    "min_samples_leaf": [1, 2, 5],
}
LGBM_VAL_GRID = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.05, 0.08, 0.1],
}


def _selected_params_from_model(model, keys: list[str]) -> dict:
    """Pulls the searched hyperparameter keys from a fitted estimator's get_params()."""
    params = model.get_params(deep=False)
    return {k: _json_safe(params.get(k)) for k in keys}


def tune_rf_on_val(
    X_tr,
    y_tr,
    X_va,
    y_va,
    *,
    task: str,
    random_state: int,
    n_estimators: int = 80,
) -> tuple[object, dict]:
    """Fits a small ParameterGrid of Random Forests on train and picks the best val macro-F1."""
    grid_axes = {
        "n_estimators": [n_estimators],
        **RF_VAL_GRID,
    }
    candidates = list(ParameterGrid(grid_axes))
    trials = []
    best_score = -1.0
    best_params = None

    for params in candidates:
        model = default_rf(task=task, random_state=random_state, n_estimators=params["n_estimators"])
        model.set_params(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = float(f1_score(y_va, preds, average="macro", zero_division=0))
        trials.append({"params": dict(params), "val_macro_f1": round(score, 4)})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    assert best_params is not None
    best_model = default_rf(task=task, random_state=random_state, n_estimators=best_params["n_estimators"])
    best_model.set_params(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )
    best_model.fit(X_tr, y_tr)
    selected = _selected_params_from_model(
        best_model, ["n_estimators", "max_depth", "min_samples_leaf", "class_weight", "max_samples"]
    )
    tuning = {
        "split": "val",
        "seed": random_state,
        "candidates_evaluated": {k: list(v) for k, v in grid_axes.items()},
        "n_trials": len(trials),
        "trials": trials,
        "selected_params": selected,
        "val_macro_f1": round(best_score, 4),
        "test_touched_during_tuning": False,
    }
    return best_model, tuning


def tune_lgbm_on_val(
    X_tr,
    y_tr,
    X_va,
    y_va,
    *,
    task: str,
    random_state: int,
    n_estimators: int = 100,
) -> tuple[object, dict]:
    """Fits a small ParameterGrid of LightGBM models on train and picks the best val macro-F1."""
    grid_axes = {
        "n_estimators": [n_estimators],
        **LGBM_VAL_GRID,
    }
    candidates = list(ParameterGrid(grid_axes))
    trials = []
    best_score = -1.0
    best_params = None

    for params in candidates:
        model = default_lgbm(task=task, random_state=random_state, n_estimators=params["n_estimators"])
        model.set_params(
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = float(f1_score(y_va, preds, average="macro", zero_division=0))
        trials.append({"params": dict(params), "val_macro_f1": round(score, 4)})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    assert best_params is not None
    best_model = default_lgbm(task=task, random_state=random_state, n_estimators=best_params["n_estimators"])
    best_model.set_params(
        num_leaves=best_params["num_leaves"],
        learning_rate=best_params["learning_rate"],
    )
    best_model.fit(X_tr, y_tr)
    selected = _selected_params_from_model(
        best_model,
        ["n_estimators", "num_leaves", "learning_rate", "min_child_samples", "class_weight", "reg_lambda", "max_delta_step"],
    )
    tuning = {
        "split": "val",
        "seed": random_state,
        "candidates_evaluated": {k: list(v) for k, v in grid_axes.items()},
        "n_trials": len(trials),
        "trials": trials,
        "selected_params": selected,
        "val_macro_f1": round(best_score, 4),
        "test_touched_during_tuning": False,
    }
    return best_model, tuning



def _cm_stats(cm: np.ndarray, labels: list[str], benign_idx: int | None) -> dict:
    """Computes macro_f1, weighted_f1, accuracy, benign_fpr, and per_class from a confusion matrix."""
    supports = cm.sum(axis=1).astype(float)
    pred_counts = cm.sum(axis=0).astype(float)
    tps = np.diag(cm).astype(float)

    precisions = np.divide(tps, pred_counts, out=np.zeros_like(tps), where=pred_counts > 0)
    recalls = np.divide(tps, supports, out=np.zeros_like(tps), where=supports > 0)
    denom = precisions + recalls
    f1s = np.divide(2.0 * precisions * recalls, denom, out=np.zeros_like(tps), where=denom > 0)

    per_class = {
        c: {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1-score": round(float(f1s[i]), 4),
            "support": round(float(supports[i]), 4),
        }
        for i, c in enumerate(labels)
    }

    total_support = float(supports.sum())
    present_mask = supports > 0
    macro_f1 = float(np.mean(f1s[present_mask])) if present_mask.any() else 0.0
    weighted_f1 = float(np.sum(f1s * supports) / total_support) if total_support > 0 else 0.0
    acc = float(tps.sum() / total_support) if total_support > 0 else 0.0

    if benign_idx is not None and supports[benign_idx] > 0:
        benign_fpr = float(1.0 - recalls[benign_idx])
    else:
        benign_fpr = None

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(acc, 4),
        "benign_fpr": round(benign_fpr, 4) if benign_fpr is not None else None,
        "per_class": per_class,
        "cm": cm,
    }


def _metrics_from_preds(
    preds: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    benign_name: str,
    feature_importances: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    twin_info: dict | None = None,
) -> dict:
    """Computes the full evaluation metrics dictionary matching eval_template.evaluate_single."""
    label_to_idx = {c: i for i, c in enumerate(labels)}
    y_int = np.fromiter((label_to_idx[v] for v in y_true), dtype=np.int32, count=len(y_true))
    p_int = np.fromiter((label_to_idx[v] for v in preds), dtype=np.int32, count=len(preds))
    labels_int = np.arange(len(labels), dtype=np.int32)
    benign_idx = label_to_idx.get(benign_name)

    full_cm = confusion_matrix(y_int, p_int, labels=labels_int)
    full_stats = _cm_stats(full_cm, labels, benign_idx)

    scores = np.asarray(feature_importances, dtype=float)
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_features = [
        {"feature": feature_names[i], "importance": round(float(scores[i]), 5)}
        for i in top_idx
    ]

    result = {
        "macro_f1": full_stats["macro_f1"],
        "weighted_f1": full_stats["weighted_f1"],
        "accuracy": full_stats["accuracy"],
        "benign_fpr": full_stats["benign_fpr"],
        "per_class": full_stats["per_class"],
        "confusion_matrix": {
            "labels": labels,
            "matrix": full_cm.tolist(),
        },
        "top_features": top_features,
    }

    if twin_info is not None:
        twin_mask = twin_info["feature_twin_mask"]
        unseen_mask = ~twin_mask
        n_cls = len(labels_int)

        unseen_cm = (
            confusion_matrix(y_int[unseen_mask], p_int[unseen_mask], labels=labels_int)
            if unseen_mask.any()
            else np.zeros((n_cls, n_cls), dtype=np.int64)
        )
        twins_cm = (
            confusion_matrix(y_int[twin_mask], p_int[twin_mask], labels=labels_int)
            if twin_mask.any()
            else np.zeros((n_cls, n_cls), dtype=np.int64)
        )
        unseen_stats = _cm_stats(unseen_cm, labels, benign_idx)
        twins_stats = _cm_stats(twins_cm, labels, benign_idx)

        per_class_twins = {}
        for i, c in enumerate(labels):
            c_total = int(full_cm[i].sum())
            c_twins = int(twins_cm[i].sum())
            c_unseen = int(unseen_cm[i].sum())
            f1_all = full_stats["per_class"][c]["f1-score"]
            f1_unseen = unseen_stats["per_class"][c]["f1-score"] if c_unseen > 0 else None
            per_class_twins[c] = {
                "test_support": c_total,
                "train_twins": c_twins,
                "unseen_support": c_unseen,
                "pct_twins": round(100.0 * c_twins / c_total, 2) if c_total > 0 else 0.0,
                "f1_all": f1_all,
                "f1_unseen": f1_unseen,
                "delta_f1_unseen_minus_all": round(f1_unseen - f1_all, 4) if f1_unseen is not None else None,
            }

        delta_f1 = round(unseen_stats["macro_f1"] - full_stats["macro_f1"], 4)
        delta_fpr = (
            round(unseen_stats["benign_fpr"] - full_stats["benign_fpr"], 4)
            if (unseen_stats["benign_fpr"] is not None and full_stats["benign_fpr"] is not None)
            else None
        )

        result["train_twin_analysis"] = {
            "n_test_rows": twin_info["n_test_rows"],
            "n_feature_twins": twin_info["n_feature_twins"],
            "pct_feature_twins": twin_info["pct_feature_twins"],
            "n_exact_twins": twin_info["n_exact_twins"],
            "pct_exact_twins": twin_info["pct_exact_twins"],
            "n_unseen_rows": twin_info["n_unseen_rows"],
            "pct_unseen_rows": twin_info["pct_unseen_rows"],
            "macro_f1_all": full_stats["macro_f1"],
            "macro_f1_twins_only": twins_stats["macro_f1"],
            "macro_f1_unseen": unseen_stats["macro_f1"],
            "delta_macro_f1_unseen_minus_all": delta_f1,
            "benign_fpr_all": full_stats["benign_fpr"],
            "benign_fpr_twins_only": twins_stats["benign_fpr"],
            "benign_fpr_unseen": unseen_stats["benign_fpr"],
            "delta_benign_fpr_unseen_minus_all": delta_fpr,
            "per_class_breakdown": per_class_twins,
        }

    return result


def _oof_meta_predictions(
    p_rf_val: np.ndarray,
    p_lgbm_val: np.ndarray,
    y_val: np.ndarray,
    classes: np.ndarray,
    meta_blueprint,
    n_splits: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Computes out-of-fold probability predictions on the validation set for unbiased meta-learner validation metrics."""
    meta_X = build_meta_features(p_rf_val, p_lgbm_val)
    _, counts = np.unique(y_val, return_counts=True)
    k_folds = min(n_splits, int(counts.min()))
    oof_probs = np.zeros((len(y_val), len(classes)), dtype=np.float32)

    if k_folds >= 2:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        for tr_idx, va_idx in skf.split(meta_X, y_val):
            if len(np.unique(y_val[tr_idx])) < len(classes):
                continue
            m_X_sub, y_sub, w_sub = _subsample_for_meta(
                meta_X[tr_idx], y_val[tr_idx], max_per_class=8000, random_state=random_state
            )
            m = clone(meta_blueprint).fit(m_X_sub, y_sub, sample_weight=w_sub)
            raw_va = m.predict_proba(meta_X[va_idx])
            oof_probs[va_idx] = 0.5 * raw_va + 0.25 * p_rf_val[va_idx] + 0.25 * p_lgbm_val[va_idx]

    missing = (oof_probs.sum(axis=1) == 0)
    if missing.any():
        m_X_sub, y_sub, w_sub = _subsample_for_meta(meta_X, y_val, max_per_class=8000, random_state=random_state)
        m_full = clone(meta_blueprint).fit(m_X_sub, y_sub, sample_weight=w_sub)
        raw_miss = m_full.predict_proba(meta_X[missing])
        oof_probs[missing] = 0.5 * raw_miss + 0.25 * p_rf_val[missing] + 0.25 * p_lgbm_val[missing]

    return oof_probs


def _save_outputs(
    out_dir: Path,
    eval_files: dict[tuple[str, str], list[dict]],
    rf_vs_lgbm_records: list[dict],
    ensemble_records: list[dict],
):
    """Writes per-model evaluation files and comparison reports to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (m_name, task), records in eval_files.items():
        path = out_dir / f"eval_{m_name}_{task}.json"
        path.write_text(json.dumps(records, indent=2))
        if m_name == "random_forest":
            (out_dir / f"eval_rf_{task}.json").write_text(json.dumps(records, indent=2))

    (out_dir / "rf_vs_lightgbm_comparison.json").write_text(json.dumps(rf_vs_lgbm_records, indent=2))
    (out_dir / "ensemble_comparison.json").write_text(json.dumps(ensemble_records, indent=2))


def run_training_and_comparison(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    use_wandb: bool = USE_WANDB,
    train_sample: int | None = None,
    n_estimators_rf: int = 80,
    n_estimators_lgbm: int = 100,
    random_state: int = SEED,
    skip_wandb_combos: set[tuple[str, bool]] | None = None,
) -> dict:
    """Trains RF and LightGBM on binary & multi-class tasks, compares them, and compares
    their Fixed-Weights Ensemble vs. Meta-Learner Ensemble with W&B tracking.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_meta = get_data_metadata(data_dir)
    commit_hash = get_code_commit()

    print("Loading train, val, and test splits...", flush=True)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    if train_sample is not None and train_sample < len(train_df):
        train_df = train_df.sample(n=train_sample, random_state=random_state)

    print(
        f"  Rows -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} "
        f"| data_version: {data_meta['data_version']} | code_commit: {commit_hash}",
        flush=True,
    )

    eval_files: dict[tuple[str, str], list[dict]] = {}
    rf_vs_lgbm_records = []
    ensemble_records = []
    wandb_summary_rows = []

    for task in ("binary", "multiclass"):
        for use_port in (True, False):
            X_tr_df, y_tr, benign_name = prep_xy(train_df, use_port, task)
            X_va_df, y_va, _ = prep_xy(val_df, use_port, task)
            X_te_df, y_te, _ = prep_xy(test_df, use_port, task)

            feature_names = list(X_tr_df.columns)
            n_features = len(feature_names)
            X_tr = X_tr_df.astype(np.float32)
            X_va = X_va_df.astype(np.float32)
            X_te = X_te_df.astype(np.float32)
            y_tr_arr = np.asarray(y_tr)
            y_va_arr = np.asarray(y_va)
            y_te_arr = np.asarray(y_te)
            # Union of train/val/test labels so a prediction outside the test-only set cannot KeyError.
            labels = sorted(
                set(pd.unique(y_tr_arr).tolist())
                | set(pd.unique(y_va_arr).tolist())
                | set(pd.unique(y_te_arr).tolist())
            )

            twin_info = find_train_twins(X_tr_df, X_te_df, y_train=y_tr_arr, y_test=y_te_arr)
            print(
                f"\n=== Task: {task} | dst_port={use_port} ({n_features} features, {len(labels)} classes) ===\n"
                f"    Train-Twin Audit: {twin_info['n_feature_twins']:,}/{twin_info['n_test_rows']:,} test rows "
                f"({twin_info['pct_feature_twins']:.2f}%) have an exact feature twin in train "
                f"({twin_info['n_unseen_rows']:,} strictly unseen test rows)",
                flush=True,
            )

            # ------------------------------------------------------------------
            # 1. Tune & Evaluate Random Forest (val-only ParameterGrid)
            # ------------------------------------------------------------------
            print(
                f"  [1/4] Tuning Random Forest on val ({len(list(ParameterGrid({'n_estimators': [n_estimators_rf], **RF_VAL_GRID})))} candidates)...",
                flush=True,
            )
            t0 = time.time()
            rf_model, rf_tuning = tune_rf_on_val(
                X_tr, y_tr_arr, X_va, y_va_arr,
                task=task, random_state=random_state, n_estimators=n_estimators_rf,
            )
            rf_fit_sec = round(time.time() - t0, 2)
            rf_params = extract_model_params(rf_model)
            rf_model_params = _model_params(rf_model)

            t0 = time.time()
            p_rf_va = rf_model.predict_proba(X_va)
            rf_val_infer_sec = round(time.time() - t0, 2)

            t0 = time.time()
            p_rf_te = rf_model.predict_proba(X_te)
            rf_te_infer_sec = round(time.time() - t0, 2)

            classes = np.asarray(rf_model.classes_)
            preds_rf_va = classes[np.argmax(p_rf_va, axis=1)]
            preds_rf_te = classes[np.argmax(p_rf_te, axis=1)]

            rf_imp = _normalize_importances(rf_model.feature_importances_)
            rf_val_metrics = _metrics_from_preds(preds_rf_va, y_va_arr, labels, benign_name, rf_imp, feature_names)
            rf_te_metrics = _metrics_from_preds(
                preds_rf_te, y_te_arr, labels, benign_name, rf_imp, feature_names, twin_info=twin_info
            )
            rf_tw = rf_te_metrics["train_twin_analysis"]
            rf_tuning.update({
                "val_benign_fpr": rf_val_metrics["benign_fpr"],
                "val_infer_time_sec": rf_val_infer_sec,
                "val_macro_f1": rf_val_metrics["macro_f1"],
            })
            rf_te_metrics.update({
                "model": "random_forest",
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "parameters": rf_params,
                "model_params": rf_model_params,
                "val_tuning": rf_tuning,
                "fit_time_sec": rf_fit_sec,
                "infer_time_sec": rf_te_infer_sec,
                "val_macro_f1": rf_val_metrics["macro_f1"],
                "val_benign_fpr": rf_val_metrics["benign_fpr"],
            })
            eval_files.setdefault(("random_forest", task), []).append(rf_te_metrics)
            print(
                f"        RF done in {rf_fit_sec:.1f}s | selected={rf_tuning['selected_params']} | "
                f"Val F1={rf_val_metrics['macro_f1']:.4f} | "
                f"Test F1(all->unseen)={rf_te_metrics['macro_f1']:.4f}->{rf_tw['macro_f1_unseen']:.4f} "
                f"(Δ={rf_tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
                f"FPR(all->unseen)={rf_te_metrics['benign_fpr']:.4f}->{rf_tw['benign_fpr_unseen']:.4f}",
                flush=True,
            )

            # ------------------------------------------------------------------
            # 2. Tune & Evaluate LightGBM (val-only ParameterGrid)
            # ------------------------------------------------------------------
            print(
                f"  [2/4] Tuning LightGBM on val ({len(list(ParameterGrid({'n_estimators': [n_estimators_lgbm], **LGBM_VAL_GRID})))} candidates)...",
                flush=True,
            )
            t0 = time.time()
            lgbm_model, lgbm_tuning = tune_lgbm_on_val(
                X_tr, y_tr_arr, X_va, y_va_arr,
                task=task, random_state=random_state, n_estimators=n_estimators_lgbm,
            )
            lgbm_fit_sec = round(time.time() - t0, 2)
            lgbm_params = extract_model_params(lgbm_model)
            lgbm_model_params = _model_params(lgbm_model)

            t0 = time.time()
            p_lgbm_va = lgbm_model.predict_proba(X_va)
            lgbm_val_infer_sec = round(time.time() - t0, 2)

            t0 = time.time()
            p_lgbm_te = lgbm_model.predict_proba(X_te)
            lgbm_te_infer_sec = round(time.time() - t0, 2)

            # Use LightGBM's own class order (same labels as RF for this data, but safer).
            lgbm_classes = np.asarray(lgbm_model.classes_)
            preds_lgbm_va = lgbm_classes[np.argmax(p_lgbm_va, axis=1)]
            preds_lgbm_te = lgbm_classes[np.argmax(p_lgbm_te, axis=1)]

            lgbm_imp = _normalize_importances(lgbm_model.feature_importances_)
            lgbm_val_metrics = _metrics_from_preds(preds_lgbm_va, y_va_arr, labels, benign_name, lgbm_imp, feature_names)
            lgbm_te_metrics = _metrics_from_preds(
                preds_lgbm_te, y_te_arr, labels, benign_name, lgbm_imp, feature_names, twin_info=twin_info
            )
            lgbm_tw = lgbm_te_metrics["train_twin_analysis"]
            lgbm_tuning.update({
                "val_benign_fpr": lgbm_val_metrics["benign_fpr"],
                "val_infer_time_sec": lgbm_val_infer_sec,
                "val_macro_f1": lgbm_val_metrics["macro_f1"],
            })
            lgbm_te_metrics.update({
                "model": "lightgbm",
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "parameters": lgbm_params,
                "model_params": lgbm_model_params,
                "val_tuning": lgbm_tuning,
                "fit_time_sec": lgbm_fit_sec,
                "infer_time_sec": lgbm_te_infer_sec,
                "val_macro_f1": lgbm_val_metrics["macro_f1"],
                "val_benign_fpr": lgbm_val_metrics["benign_fpr"],
            })
            eval_files.setdefault(("lightgbm", task), []).append(lgbm_te_metrics)
            print(
                f"        LightGBM done in {lgbm_fit_sec:.1f}s | selected={lgbm_tuning['selected_params']} | "
                f"Val F1={lgbm_val_metrics['macro_f1']:.4f} | "
                f"Test F1(all->unseen)={lgbm_te_metrics['macro_f1']:.4f}->{lgbm_tw['macro_f1_unseen']:.4f} "
                f"(Δ={lgbm_tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
                f"FPR(all->unseen)={lgbm_te_metrics['benign_fpr']:.4f}->{lgbm_tw['benign_fpr_unseen']:.4f}",
                flush=True,
            )

            # ------------------------------------------------------------------
            # 3. Fixed-Weights Ensemble (Equal 0.5/0.5 + Validation-Sweep Tuned)
            # ------------------------------------------------------------------
            print("  [3/4] Evaluating Fixed-Weights Ensemble (0.5/0.5 and validation weight sweep)...", flush=True)
            fixed_ens = FixedWeightEnsemble.from_prefitted(rf_model, lgbm_model, weights=(0.5, 0.5), task=task)
            fixed_params = extract_model_params(fixed_ens)
            fixed_model_params = _model_params(fixed_ens)
            p_fixed_va = fixed_ens.predict_proba_from_base(p_rf_va, p_lgbm_va)
            t0 = time.time()
            p_fixed_te = fixed_ens.predict_proba_from_base(p_rf_te, p_lgbm_te)
            fixed_te_infer_sec = round(rf_te_infer_sec + lgbm_te_infer_sec + (time.time() - t0), 2)

            preds_fixed_va = classes[np.argmax(p_fixed_va, axis=1)]
            preds_fixed_te = classes[np.argmax(p_fixed_te, axis=1)]

            fixed_val_metrics = _metrics_from_preds(
                preds_fixed_va, y_va_arr, labels, benign_name, fixed_ens.feature_importances_, feature_names
            )
            fixed_te_metrics = _metrics_from_preds(
                preds_fixed_te, y_te_arr, labels, benign_name, fixed_ens.feature_importances_, feature_names, twin_info=twin_info
            )
            fixed_tw = fixed_te_metrics["train_twin_analysis"]

            sweep_info = sweep_fixed_weights(p_rf_va, p_lgbm_va, y_va_arr, classes, benign_name, steps=21)
            best_w = sweep_info["best_weights"]
            fixed_tuned_ens = FixedWeightEnsemble.from_prefitted(rf_model, lgbm_model, weights=best_w, task=task)
            fixed_tuned_params = extract_model_params(fixed_tuned_ens)
            fixed_tuned_model_params = _model_params(fixed_tuned_ens)
            p_tuned_te = fixed_tuned_ens.predict_proba_from_base(p_rf_te, p_lgbm_te)
            preds_tuned_te = classes[np.argmax(p_tuned_te, axis=1)]
            fixed_tuned_te_metrics = _metrics_from_preds(
                preds_tuned_te, y_te_arr, labels, benign_name, fixed_tuned_ens.feature_importances_, feature_names, twin_info=twin_info
            )
            tuned_tw = fixed_tuned_te_metrics["train_twin_analysis"]

            fixed_te_metrics.update({
                "model": "ensemble_fixed",
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "parameters": fixed_params,
                "model_params": fixed_model_params,
                "weights": {"rf": 0.5, "lgbm": 0.5},
                "val_tuned_weights": {"rf": best_w[0], "lgbm": best_w[1]},
                "val_tuned_test_macro_f1": fixed_tuned_te_metrics["macro_f1"],
                "val_tuned_test_macro_f1_unseen": tuned_tw["macro_f1_unseen"],
                "val_tuned_test_benign_fpr": fixed_tuned_te_metrics["benign_fpr"],
                "val_tuned_test_benign_fpr_unseen": tuned_tw["benign_fpr_unseen"],
                "val_tuning": {
                    "split": "val",
                    "seed": random_state,
                    "sweep_steps": 21,
                    "best_weights": {"rf": best_w[0], "lgbm": best_w[1]},
                    "val_macro_f1": round(sweep_info["best_macro_f1"], 4),
                    "val_benign_fpr": round(sweep_info["best_benign_fpr"], 4),
                    "test_touched_during_tuning": False,
                },
                "fit_time_sec": round(rf_fit_sec + lgbm_fit_sec, 2),
                "infer_time_sec": fixed_te_infer_sec,
                "val_macro_f1": fixed_val_metrics["macro_f1"],
                "val_benign_fpr": fixed_val_metrics["benign_fpr"],
            })
            fixed_tuned_te_metrics.update({
                "model": "ensemble_fixed_tuned",
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "parameters": fixed_tuned_params,
                "model_params": fixed_tuned_model_params,
                "fit_time_sec": round(rf_fit_sec + lgbm_fit_sec, 2),
                "infer_time_sec": fixed_te_infer_sec,
            })
            eval_files.setdefault(("ensemble_fixed", task), []).append(fixed_te_metrics)
            print(
                f"        Fixed (0.5/0.5)   | Val F1={fixed_val_metrics['macro_f1']:.4f} | "
                f"Test F1(all->unseen)={fixed_te_metrics['macro_f1']:.4f}->{fixed_tw['macro_f1_unseen']:.4f} "
                f"(Δ={fixed_tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
                f"FPR(all->unseen)={fixed_te_metrics['benign_fpr']:.4f}->{fixed_tw['benign_fpr_unseen']:.4f}",
                flush=True,
            )
            print(
                f"        Fixed (Tuned {best_w[0]:.2f}/{best_w[1]:.2f}) | Val F1={sweep_info['best_macro_f1']:.4f} | "
                f"Test F1(all->unseen)={fixed_tuned_te_metrics['macro_f1']:.4f}->{tuned_tw['macro_f1_unseen']:.4f} "
                f"(Δ={tuned_tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
                f"FPR(all->unseen)={fixed_tuned_te_metrics['benign_fpr']:.4f}->{tuned_tw['benign_fpr_unseen']:.4f}",
                flush=True,
            )

            # ------------------------------------------------------------------
            # 4. Meta-Learner Ensemble (Stacking LogisticRegression)
            # ------------------------------------------------------------------
            print("  [4/4] Training & Evaluating Meta-Learner (Stacking) Ensemble...", flush=True)
            t0 = time.time()
            meta_ens = MetaLearnerEnsemble.from_prefitted(
                rf_model,
                lgbm_model,
                X_va,
                y_va_arr,
                task=task,
                random_state=random_state,
                p_rf_val=p_rf_va,
                p_lgbm_val=p_lgbm_va,
            )
            meta_only_fit_sec = round(time.time() - t0, 2)
            meta_params = extract_model_params(meta_ens)
            meta_model_params = _model_params(meta_ens)

            p_meta_va_oof = _oof_meta_predictions(
                p_rf_va, p_lgbm_va, y_va_arr, classes, meta_ens._make_meta_learner(), n_splits=3, random_state=random_state
            )
            preds_meta_va = classes[np.argmax(p_meta_va_oof * meta_ens.class_multipliers_, axis=1)]

            t0 = time.time()
            p_meta_te = meta_ens.predict_proba_from_base(p_rf_te, p_lgbm_te)
            meta_te_infer_sec = round(rf_te_infer_sec + lgbm_te_infer_sec + (time.time() - t0), 2)
            preds_meta_te = classes[np.argmax(p_meta_te, axis=1)]

            meta_contrib = meta_ens.get_model_contributions()
            meta_val_metrics = _metrics_from_preds(
                preds_meta_va, y_va_arr, labels, benign_name, meta_ens.feature_importances_, feature_names
            )
            meta_te_metrics = _metrics_from_preds(
                preds_meta_te, y_te_arr, labels, benign_name, meta_ens.feature_importances_, feature_names, twin_info=twin_info
            )
            meta_tw = meta_te_metrics["train_twin_analysis"]
            meta_te_metrics.update({
                "model": "ensemble_metalearner",
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "parameters": meta_params,
                "model_params": meta_model_params,
                "meta_learner_type": "LogisticRegression(C=20.0, features=[prob, scaled_log_prob, abs_diff]) + class_multipliers",
                "learned_model_contributions": meta_contrib,
                "val_tuning": {
                    "split": "val",
                    "seed": random_state,
                    "oof_folds": 3,
                    "meta_learner": "LogisticRegression(C=20.0) + coordinate-ascent class_multipliers on val OOF",
                    "val_oof_macro_f1": meta_val_metrics["macro_f1"],
                    "val_oof_benign_fpr": meta_val_metrics["benign_fpr"],
                    "test_touched_during_tuning": False,
                },
                "meta_fit_time_sec": meta_only_fit_sec,
                "fit_time_sec": round(rf_fit_sec + lgbm_fit_sec + meta_only_fit_sec, 2),
                "infer_time_sec": meta_te_infer_sec,
                "val_macro_f1": meta_val_metrics["macro_f1"],
                "val_benign_fpr": meta_val_metrics["benign_fpr"],
            })
            eval_files.setdefault(("ensemble_metalearner", task), []).append(meta_te_metrics)
            print(
                f"        Meta-Learner      | Val (OOF) F1={meta_val_metrics['macro_f1']:.4f} | "
                f"Test F1(all->unseen)={meta_te_metrics['macro_f1']:.4f}->{meta_tw['macro_f1_unseen']:.4f} "
                f"(Δ={meta_tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
                f"FPR(all->unseen)={meta_te_metrics['benign_fpr']:.4f}->{meta_tw['benign_fpr_unseen']:.4f} | "
                f"Shares: RF={meta_contrib['rf_share']:.2f}, LGBM={meta_contrib['lgbm_share']:.2f}",
                flush=True,
            )

            # ------------------------------------------------------------------
            # Build Comparison Records
            # ------------------------------------------------------------------
            per_class_rf_vs_lgbm = {}
            per_class_meta_vs_fixed = {}
            for c in labels:
                rf_c_f1 = rf_te_metrics["per_class"][c]["f1-score"]
                lgbm_c_f1 = lgbm_te_metrics["per_class"][c]["f1-score"]
                fixed_c_f1 = fixed_te_metrics["per_class"][c]["f1-score"]
                tuned_c_f1 = fixed_tuned_te_metrics["per_class"][c]["f1-score"]
                meta_c_f1 = meta_te_metrics["per_class"][c]["f1-score"]
                support_c = int(rf_te_metrics["per_class"][c]["support"])

                per_class_rf_vs_lgbm[c] = {
                    "support": support_c,
                    "unseen_support": rf_tw["per_class_breakdown"][c]["unseen_support"],
                    "pct_twins": rf_tw["per_class_breakdown"][c]["pct_twins"],
                    "rf_f1": rf_c_f1,
                    "rf_f1_unseen": rf_tw["per_class_breakdown"][c]["f1_unseen"],
                    "lgbm_f1": lgbm_c_f1,
                    "lgbm_f1_unseen": lgbm_tw["per_class_breakdown"][c]["f1_unseen"],
                    "delta_rf_minus_lgbm": round(rf_c_f1 - lgbm_c_f1, 4),
                    "winner": "random_forest" if rf_c_f1 > lgbm_c_f1 else ("lightgbm" if lgbm_c_f1 > rf_c_f1 else "tie"),
                }
                per_class_meta_vs_fixed[c] = {
                    "support": support_c,
                    "unseen_support": meta_tw["per_class_breakdown"][c]["unseen_support"],
                    "pct_twins": meta_tw["per_class_breakdown"][c]["pct_twins"],
                    "fixed_05_05_f1": fixed_c_f1,
                    "fixed_05_05_f1_unseen": fixed_tw["per_class_breakdown"][c]["f1_unseen"],
                    "fixed_tuned_f1": tuned_c_f1,
                    "fixed_tuned_f1_unseen": tuned_tw["per_class_breakdown"][c]["f1_unseen"],
                    "metalearner_f1": meta_c_f1,
                    "metalearner_f1_unseen": meta_tw["per_class_breakdown"][c]["f1_unseen"],
                    "delta_meta_minus_fixed_05_05": round(meta_c_f1 - fixed_c_f1, 4),
                    "delta_meta_minus_fixed_tuned": round(meta_c_f1 - tuned_c_f1, 4),
                }

            rf_top5 = {f["feature"] for f in rf_te_metrics["top_features"][:5]}
            lgbm_top5 = {f["feature"] for f in lgbm_te_metrics["top_features"][:5]}

            rf_vs_lgbm_records.append({
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "train_twin_summary": {
                    "n_test_rows": twin_info["n_test_rows"],
                    "n_feature_twins": twin_info["n_feature_twins"],
                    "pct_feature_twins": twin_info["pct_feature_twins"],
                    "n_unseen_rows": twin_info["n_unseen_rows"],
                },
                "random_forest": {
                    "parameters": rf_params,
                    "val_macro_f1": rf_val_metrics["macro_f1"],
                    "val_benign_fpr": rf_val_metrics["benign_fpr"],
                    "test_macro_f1": rf_te_metrics["macro_f1"],
                    "test_macro_f1_unseen": rf_tw["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": rf_tw["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": rf_te_metrics["weighted_f1"],
                    "test_accuracy": rf_te_metrics["accuracy"],
                    "test_benign_fpr": rf_te_metrics["benign_fpr"],
                    "test_benign_fpr_unseen": rf_tw["benign_fpr_unseen"],
                    "fit_time_sec": rf_fit_sec,
                    "test_infer_time_sec": rf_te_infer_sec,
                    "top_5_features": rf_te_metrics["top_features"][:5],
                },
                "lightgbm": {
                    "parameters": lgbm_params,
                    "val_macro_f1": lgbm_val_metrics["macro_f1"],
                    "val_benign_fpr": lgbm_val_metrics["benign_fpr"],
                    "test_macro_f1": lgbm_te_metrics["macro_f1"],
                    "test_macro_f1_unseen": lgbm_tw["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": lgbm_tw["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": lgbm_te_metrics["weighted_f1"],
                    "test_accuracy": lgbm_te_metrics["accuracy"],
                    "test_benign_fpr": lgbm_te_metrics["benign_fpr"],
                    "test_benign_fpr_unseen": lgbm_tw["benign_fpr_unseen"],
                    "fit_time_sec": lgbm_fit_sec,
                    "test_infer_time_sec": lgbm_te_infer_sec,
                    "top_5_features": lgbm_te_metrics["top_features"][:5],
                },
                "delta_rf_minus_lgbm": {
                    "val_macro_f1": round(rf_val_metrics["macro_f1"] - lgbm_val_metrics["macro_f1"], 4),
                    "test_macro_f1": round(rf_te_metrics["macro_f1"] - lgbm_te_metrics["macro_f1"], 4),
                    "test_macro_f1_unseen": round(rf_tw["macro_f1_unseen"] - lgbm_tw["macro_f1_unseen"], 4),
                    "test_benign_fpr": round(rf_te_metrics["benign_fpr"] - lgbm_te_metrics["benign_fpr"], 4),
                    "fit_time_sec": round(rf_fit_sec - lgbm_fit_sec, 2),
                },
                "winner_test_macro_f1": (
                    "random_forest"
                    if rf_te_metrics["macro_f1"] > lgbm_te_metrics["macro_f1"]
                    else ("lightgbm" if lgbm_te_metrics["macro_f1"] > rf_te_metrics["macro_f1"] else "tie")
                ),
                "top_5_feature_overlap": sorted(rf_top5 & lgbm_top5),
                "per_class_comparison": per_class_rf_vs_lgbm,
            })

            ensemble_records.append({
                "task": task,
                "dst_port": use_port,
                "n_features": n_features,
                "seed": random_state,
                "data_version": data_meta["data_version"],
                "code_commit": commit_hash,
                "train_twin_summary": {
                    "n_test_rows": twin_info["n_test_rows"],
                    "n_feature_twins": twin_info["n_feature_twins"],
                    "pct_feature_twins": twin_info["pct_feature_twins"],
                    "n_unseen_rows": twin_info["n_unseen_rows"],
                },
                "base_models": {
                    "random_forest_test_macro_f1": rf_te_metrics["macro_f1"],
                    "random_forest_test_macro_f1_unseen": rf_tw["macro_f1_unseen"],
                    "random_forest_test_benign_fpr": rf_te_metrics["benign_fpr"],
                    "lightgbm_test_macro_f1": lgbm_te_metrics["macro_f1"],
                    "lightgbm_test_macro_f1_unseen": lgbm_tw["macro_f1_unseen"],
                    "lightgbm_test_benign_fpr": lgbm_te_metrics["benign_fpr"],
                },
                "fixed_weights_equal_05_05": {
                    "weights": {"rf": 0.5, "lgbm": 0.5},
                    "parameters": fixed_params,
                    "val_macro_f1": fixed_val_metrics["macro_f1"],
                    "val_benign_fpr": fixed_val_metrics["benign_fpr"],
                    "test_macro_f1": fixed_te_metrics["macro_f1"],
                    "test_macro_f1_unseen": fixed_tw["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": fixed_tw["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": fixed_te_metrics["weighted_f1"],
                    "test_accuracy": fixed_te_metrics["accuracy"],
                    "test_benign_fpr": fixed_te_metrics["benign_fpr"],
                    "test_benign_fpr_unseen": fixed_tw["benign_fpr_unseen"],
                },
                "fixed_weights_val_tuned": {
                    "weights": {"rf": best_w[0], "lgbm": best_w[1]},
                    "parameters": fixed_tuned_params,
                    "val_macro_f1": round(sweep_info["best_macro_f1"], 4),
                    "val_benign_fpr": round(sweep_info["best_benign_fpr"], 4),
                    "test_macro_f1": fixed_tuned_te_metrics["macro_f1"],
                    "test_macro_f1_unseen": tuned_tw["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": tuned_tw["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": fixed_tuned_te_metrics["weighted_f1"],
                    "test_accuracy": fixed_tuned_te_metrics["accuracy"],
                    "test_benign_fpr": fixed_tuned_te_metrics["benign_fpr"],
                    "test_benign_fpr_unseen": tuned_tw["benign_fpr_unseen"],
                },
                "metalearner_stacking": {
                    "meta_learner": "LogisticRegression(C=20.0) + per-class Macro-F1 multipliers",
                    "parameters": meta_params,
                    "learned_contributions": meta_contrib,
                    "val_oof_macro_f1": meta_val_metrics["macro_f1"],
                    "val_oof_benign_fpr": meta_val_metrics["benign_fpr"],
                    "test_macro_f1": meta_te_metrics["macro_f1"],
                    "test_macro_f1_unseen": meta_tw["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": meta_tw["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": meta_te_metrics["weighted_f1"],
                    "test_accuracy": meta_te_metrics["accuracy"],
                    "test_benign_fpr": meta_te_metrics["benign_fpr"],
                    "test_benign_fpr_unseen": meta_tw["benign_fpr_unseen"],
                    "meta_fit_time_sec": meta_only_fit_sec,
                },
                "comparison_deltas": {
                    "metalearner_minus_fixed_equal_test_macro_f1": round(
                        meta_te_metrics["macro_f1"] - fixed_te_metrics["macro_f1"], 4
                    ),
                    "metalearner_minus_fixed_equal_test_macro_f1_unseen": round(
                        meta_tw["macro_f1_unseen"] - fixed_tw["macro_f1_unseen"], 4
                    ),
                    "metalearner_minus_fixed_tuned_test_macro_f1": round(
                        meta_te_metrics["macro_f1"] - fixed_tuned_te_metrics["macro_f1"], 4
                    ),
                    "metalearner_minus_fixed_tuned_test_macro_f1_unseen": round(
                        meta_tw["macro_f1_unseen"] - tuned_tw["macro_f1_unseen"], 4
                    ),
                    "metalearner_minus_best_single_test_macro_f1": round(
                        meta_te_metrics["macro_f1"] - max(rf_te_metrics["macro_f1"], lgbm_te_metrics["macro_f1"]), 4
                    ),
                    "fixed_equal_minus_best_single_test_macro_f1": round(
                        fixed_te_metrics["macro_f1"] - max(rf_te_metrics["macro_f1"], lgbm_te_metrics["macro_f1"]), 4
                    ),
                },
                "winner_ensemble_test_macro_f1": (
                    "metalearner"
                    if meta_te_metrics["macro_f1"] > fixed_te_metrics["macro_f1"]
                    else ("fixed_weights" if fixed_te_metrics["macro_f1"] > meta_te_metrics["macro_f1"] else "tie")
                ),
                "fixed_weight_val_sweep": sweep_info["grid"],
                "per_class_comparison": per_class_meta_vs_fixed,
            })

            # Save incremental JSON outputs immediately after each combination
            _save_outputs(out_dir, eval_files, rf_vs_lgbm_records, ensemble_records)

            run_items = [
                ("random_forest", rf_val_metrics, rf_te_metrics, rf_params),
                ("lightgbm", lgbm_val_metrics, lgbm_te_metrics, lgbm_params),
                ("ensemble_fixed", fixed_val_metrics, fixed_te_metrics, {"w_rf": 0.5, "w_lgbm": 0.5, **fixed_params}),
                (
                    "ensemble_fixed_tuned",
                    {"macro_f1": round(sweep_info["best_macro_f1"], 4), "benign_fpr": round(sweep_info["best_benign_fpr"], 4)},
                    fixed_tuned_te_metrics,
                    {"w_rf": best_w[0], "w_lgbm": best_w[1], **fixed_tuned_params},
                ),
                ("ensemble_metalearner", meta_val_metrics, meta_te_metrics, {**meta_contrib, **meta_params}),
            ]
            for m_name, m_val, m_te, extra_cfg in run_items:
                tw_m = m_te["train_twin_analysis"]
                wandb_summary_rows.append({
                    "model": m_name,
                    "task": task,
                    "dst_port": use_port,
                    "n_features": n_features,
                    "seed": random_state,
                    "data_version": data_meta["data_version"],
                    "code_commit": commit_hash,
                    "pct_feature_twins": tw_m["pct_feature_twins"],
                    "val_macro_f1": m_val["macro_f1"],
                    "val_benign_fpr": m_val["benign_fpr"],
                    "test_macro_f1": m_te["macro_f1"],
                    "test_macro_f1_unseen": tw_m["macro_f1_unseen"],
                    "delta_macro_f1_unseen_minus_all": tw_m["delta_macro_f1_unseen_minus_all"],
                    "test_weighted_f1": m_te["weighted_f1"],
                    "test_accuracy": m_te["accuracy"],
                    "test_benign_fpr": m_te["benign_fpr"],
                    "test_benign_fpr_unseen": tw_m["benign_fpr_unseen"],
                })

            if use_wandb and (skip_wandb_combos is None or (task, use_port) not in skip_wandb_combos):
                import wandb
                for m_name, m_val, m_te, extra_cfg in run_items:
                    tw_m = m_te["train_twin_analysis"]
                    run_name = f"{m_name}_{task}_{'port' if use_port else 'noport'}"
                    cfg = {
                        "model": m_name,
                        "task": task,
                        "dst_port": use_port,
                        "n_features": n_features,
                        "seed": random_state,
                        "data_version": data_meta["data_version"],
                        "code_commit": commit_hash,
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                        "parameters": extra_cfg,
                        **data_meta,
                    }
                    with wandb.init(**_wandb_init_kwargs(job_type="train_eval", name=run_name, config=cfg)) as run:
                        log_payload = {
                            "macro_f1": m_te["macro_f1"],
                            "weighted_f1": m_te["weighted_f1"],
                            "accuracy": m_te["accuracy"],
                            "benign_fpr": m_te["benign_fpr"],
                            "macro_f1_twins_only": tw_m["macro_f1_twins_only"],
                            "macro_f1_unseen": tw_m["macro_f1_unseen"],
                            "benign_fpr_unseen": tw_m["benign_fpr_unseen"],
                            "delta_macro_f1_unseen_minus_all": tw_m["delta_macro_f1_unseen_minus_all"],
                            "delta_benign_fpr_unseen_minus_all": tw_m["delta_benign_fpr_unseen_minus_all"],
                            "n_feature_twins": tw_m["n_feature_twins"],
                            "pct_feature_twins": tw_m["pct_feature_twins"],
                            "n_exact_twins": tw_m["n_exact_twins"],
                            "pct_exact_twins": tw_m["pct_exact_twins"],
                            "n_unseen_rows": tw_m["n_unseen_rows"],
                            "val/macro_f1": m_val["macro_f1"],
                            "val/benign_fpr": m_val["benign_fpr"],
                            "test/macro_f1": m_te["macro_f1"],
                            "test/macro_f1_unseen": tw_m["macro_f1_unseen"],
                            "test/weighted_f1": m_te["weighted_f1"],
                            "test/accuracy": m_te["accuracy"],
                            "test/benign_fpr": m_te["benign_fpr"],
                            "test/benign_fpr_unseen": tw_m["benign_fpr_unseen"],
                            "fit_time_sec": m_te.get("fit_time_sec", 0.0),
                            "infer_time_sec": m_te.get("infer_time_sec", 0.0),
                        }
                        for c_label, c_stats in m_te["per_class"].items():
                            log_payload[f"test_f1/{c_label}"] = c_stats["f1-score"]
                            log_payload[f"test_recall/{c_label}"] = c_stats["recall"]
                            log_payload[f"test_precision/{c_label}"] = c_stats["precision"]
                        for c_label, c_tw in tw_m["per_class_breakdown"].items():
                            if c_tw["f1_unseen"] is not None:
                                log_payload[f"test_f1_unseen/{c_label}"] = c_tw["f1_unseen"]
                        if m_te.get("top_features"):
                            feat_table = wandb.Table(
                                columns=["rank", "feature", "importance"],
                                data=[[idx + 1, f["feature"], f["importance"]] for idx, f in enumerate(m_te["top_features"])],
                            )
                            log_payload["top_features_table"] = feat_table
                        run.log(log_payload)

    _save_outputs(out_dir, eval_files, rf_vs_lgbm_records, ensemble_records)
    print(f"\nAll evaluation and comparison JSON reports saved to {out_dir}", flush=True)

    if use_wandb:
        import wandb
        with wandb.init(
            **_wandb_init_kwargs(
                job_type="comparison_summary",
                name="rf_lgbm_ensemble_comparison_summary",
                config={
                    "models": ["random_forest", "lightgbm", "ensemble_fixed", "ensemble_fixed_tuned", "ensemble_metalearner"],
                    "seed": random_state,
                    "data_version": data_meta["data_version"],
                    "code_commit": commit_hash,
                    **data_meta,
                },
            )
        ) as run:
            cols = [
                "model", "task", "dst_port", "n_features", "seed",
                "data_version", "code_commit", "pct_feature_twins",
                "val_macro_f1", "val_benign_fpr",
                "test_macro_f1", "test_macro_f1_unseen", "delta_macro_f1_unseen_minus_all",
                "test_weighted_f1", "test_accuracy", "test_benign_fpr", "test_benign_fpr_unseen",
            ]
            table = wandb.Table(columns=cols, data=[[row[c] for c in cols] for row in wandb_summary_rows])
            run.log({"comparison_table": table})

    return {
        "eval_files": {f"{m}_{t}": recs for (m, t), recs in eval_files.items()},
        "rf_vs_lightgbm": rf_vs_lgbm_records,
        "ensemble_comparison": ensemble_records,
        "summary_rows": wandb_summary_rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--train-sample", type=int, default=None, help="Optional subsample of train rows")
    p.add_argument("--rf-trees", type=int, default=80)
    p.add_argument("--lgbm-trees", type=int, default=100)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--skip-logged-wandb",
        action="store_true",
        help="Skip W&B logging for the first 3 combos if already synced in this session",
    )
    args = p.parse_args()

    skip_set = {("binary", True), ("binary", False), ("multiclass", True)} if args.skip_logged_wandb else None

    run_training_and_comparison(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        use_wandb=not args.no_wandb,
        train_sample=args.train_sample,
        n_estimators_rf=args.rf_trees,
        n_estimators_lgbm=args.lgbm_trees,
        random_state=args.seed,
        skip_wandb_combos=skip_set,
    )


if __name__ == "__main__":
    main()

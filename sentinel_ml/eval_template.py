import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SEED = 42

# Identifier and metadata columns excluded to prevent data leakage (SEN-FR-03)
ID_COLS = [
    "id", "expiration_id", "flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms", "dst2src_first_seen_ms",
    "dst2src_last_seen_ms", "timestamp", "label"
]


USE_WANDB = True
WANDB_ENTITY = "wcs-sentinel"
WANDB_PROJECT = "Sentinel"
DEFAULT_DATA_VERSION = "nfs2023te-v1"


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _data_version(data_dir: Path) -> str:
    card = Path(data_dir) / "data_card.json"
    if not card.exists():
        return DEFAULT_DATA_VERSION
    try:
        return json.loads(card.read_text()).get("version", DEFAULT_DATA_VERSION)
    except Exception:
        return DEFAULT_DATA_VERSION


def get_code_commit() -> str:
    """Returns the current git commit hash (with -dirty if modified), or 'unknown' outside git."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return _code_commit()


def get_data_metadata(data_dir: Path = Path("data/processed")) -> dict:
    """Reads data_version, code_commit, split row counts, and SHA-256 hashes from data_card.json if present."""
    card_path = Path(data_dir) / "data_card.json"
    card = {}
    if card_path.exists():
        try:
            card = json.loads(card_path.read_text())
        except Exception:
            card = {}

    files_info = card.get("files", {})
    return {
        "data_version": card.get("version", DEFAULT_DATA_VERSION),
        "data_source": card.get("source", "NFS-2023-TE"),
        "data_split_rule": card.get("split", "time-based 60/20/20 inside each (day, label) window"),
        "gap_seconds": card.get("gap_seconds", 1),
        "data_card_commit": card.get("code_commit"),
        "train_sha256": files_info.get("train.parquet", {}).get("sha256"),
        "val_sha256": files_info.get("val.parquet", {}).get("sha256"),
        "test_sha256": files_info.get("test.parquet", {}).get("sha256"),
    }


def extract_model_params(model) -> dict:
    """Extracts JSON/W&B-safe hyperparameters from an estimator or Pipeline."""
    raw = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    clean = {}
    for k, v in raw.items():
        if k == "steps" or callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            clean[k] = list(v)
        elif isinstance(v, dict) and all(isinstance(val, (str, int, float, bool)) or val is None for val in v.values()):
            clean[k] = {str(dk): dv for dk, dv in v.items()}
        else:
            clean[k] = type(v).__name__
    return clean


def get_wandb_init_kwargs(job_type: str, name: str, config: dict) -> dict:
    """Builds wandb.init arguments, using online mode when authenticated and offline fallback otherwise."""
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        os.environ.setdefault("WANDB_DIR", "/tmp")
    if "WANDB_MODE" in os.environ:
        mode = os.environ["WANDB_MODE"]
    elif "PYTEST_CURRENT_TEST" in os.environ:
        mode = "offline"
    else:
        has_key = bool(os.environ.get("WANDB_API_KEY"))
        if not has_key:
            try:
                import wandb
                has_key = bool(wandb.login(prompt=False))
            except Exception:
                has_key = False
        mode = "online" if has_key else "offline"

    kwargs = {
        "project": WANDB_PROJECT,
        "name": name,
        "job_type": job_type,
        "config": config,
        "mode": mode,
        "reinit": "finish_previous",
    }
    if mode == "online":
        kwargs["entity"] = WANDB_ENTITY
    return kwargs


def prep_xy(df: pd.DataFrame, use_port: bool, task: str, benign_raw: str = "BENIGN"):
    """Splits a dataframe into numeric features (X) and target labels (y)."""
    drop_cols = ID_COLS + ([] if use_port else ["dst_port"])
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
    y = np.where(df["label"] == benign_raw, "benign", "attack") if task == "binary" else df["label"].astype(str)
    benign_name = "benign" if task == "binary" else benign_raw
    return X, y, benign_name


def get_top_features(model, X_test: pd.DataFrame, y_test, top_k: int = 10) -> list[dict]:
    """Extracts top features from Trees, Linear models, or Pipelines (with Permutation fallback)."""
    # Unwrap sklearn Pipeline if needed to inspect the final estimator
    est = model.steps[-1][1] if hasattr(model, "steps") else model

    if hasattr(est, "feature_importances_"):          # Random Forest, XGBoost, LightGBM
        scores = est.feature_importances_
    elif hasattr(est, "coef_"):                       # LogisticRegression, LinearSVC
        scores = np.mean(np.abs(est.coef_), axis=0) if est.coef_.ndim > 1 else np.abs(est.coef_[0])
    else:                                             # Fallback for MLP / KNN / SVM
        # permutation_importance is O(n_features * n_repeats * predict_cost); on
        # the full 630k-row test set this dominates multiclass eval (hours). Rank
        # order stabilises with much less, so cap at 50k rows for the importance
        # computation only. Main metrics (macro-F1, FPR, per-class, CM) above are
        # still computed on the full test set.
        PERM_MAX_ROWS = 50_000
        if len(X_test) > PERM_MAX_ROWS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_test), size=PERM_MAX_ROWS, replace=False)
            X_perm = X_test.iloc[idx]
            y_perm = np.asarray(y_test)[idx]
        else:
            X_perm, y_perm = X_test, y_test
        perm = permutation_importance(model, X_perm, y_perm, n_repeats=3, random_state=42, scoring="f1_macro")
        scores = perm.importances_mean

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": X_test.columns[i], "importance": round(float(scores[i]), 5)} for i in top_idx]


def find_train_twins(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train=None,
    y_test=None,
) -> dict:
    """Identifies test rows whose feature vector (and optionally label) has an exact twin in train."""
    tr_f_hash = pd.util.hash_pandas_object(X_train, index=False)
    te_f_hash = pd.util.hash_pandas_object(X_test, index=False)
    feature_twin_mask = te_f_hash.isin(set(tr_f_hash)).to_numpy(dtype=bool)

    if y_train is not None and y_test is not None:
        tr_fl = X_train.assign(_target_label=np.asarray(y_train, dtype=str))
        te_fl = X_test.assign(_target_label=np.asarray(y_test, dtype=str))
        tr_fl_hash = pd.util.hash_pandas_object(tr_fl, index=False)
        te_fl_hash = pd.util.hash_pandas_object(te_fl, index=False)
        exact_twin_mask = te_fl_hash.isin(set(tr_fl_hash)).to_numpy(dtype=bool)
    else:
        exact_twin_mask = feature_twin_mask

    n_test = int(len(X_test))
    n_feature_twins = int(feature_twin_mask.sum())
    n_exact_twins = int(exact_twin_mask.sum())
    n_unseen = int((~feature_twin_mask).sum())

    return {
        "feature_twin_mask": feature_twin_mask,
        "exact_twin_mask": exact_twin_mask,
        "n_test_rows": n_test,
        "n_feature_twins": n_feature_twins,
        "pct_feature_twins": round(100.0 * n_feature_twins / n_test, 2) if n_test > 0 else 0.0,
        "n_exact_twins": n_exact_twins,
        "pct_exact_twins": round(100.0 * n_exact_twins / n_test, 2) if n_test > 0 else 0.0,
        "n_unseen_rows": n_unseen,
        "pct_unseen_rows": round(100.0 * n_unseen / n_test, 2) if n_test > 0 else 0.0,
    }


def _compute_subset_metrics(preds: np.ndarray, y_true: np.ndarray, labels: list[str], benign_name: str) -> dict:
    """Computes macro_f1, weighted_f1, accuracy, benign_fpr, and per_class dictionary on a given subset of predictions."""
    if len(y_true) == 0:
        return {"macro_f1": None, "weighted_f1": None, "accuracy": None, "benign_fpr": None, "per_class": {}}

    present_labels = [c for c in labels if c in set(pd.unique(y_true))]
    eval_labels = present_labels if present_labels else labels

    report = classification_report(y_true, preds, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
        for c in labels
    }

    is_benign = (y_true == benign_name)
    benign_fpr = float((preds[is_benign] != benign_name).mean()) if is_benign.any() else None
    macro_f1 = float(f1_score(y_true, preds, labels=eval_labels, average="macro", zero_division=0))
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))
    acc = float((preds == y_true).mean())

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(acc, 4),
        "benign_fpr": round(benign_fpr, 4) if benign_fpr is not None else None,
        "per_class": per_class,
    }


def evaluate_single(
    model,
    X_test: pd.DataFrame,
    y_test,
    benign_name: str,
    top_k: int = 10,
    X_train: pd.DataFrame | None = None,
    y_train=None,
) -> dict:
    """Computes all required metrics on a single fitted model and test split, plus train-twin deduplicated scores when X_train is provided."""
    preds = np.asarray(model.predict(X_test))
    y_test_arr = np.asarray(y_test)
    labels = sorted(pd.unique(y_test_arr))

    full_stats = _compute_subset_metrics(preds, y_test_arr, labels, benign_name)

    result = {
        "macro_f1": full_stats["macro_f1"],
        "weighted_f1": full_stats["weighted_f1"],
        "accuracy": full_stats["accuracy"],
        "benign_fpr": full_stats["benign_fpr"],
        "per_class": full_stats["per_class"],
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test_arr, preds, labels=labels).tolist(),
        },
        "top_features": get_top_features(model, X_test, y_test_arr, top_k=top_k),
        "parameters": extract_model_params(model),
    }

    if X_train is not None:
        twin_info = find_train_twins(X_train, X_test, y_train=y_train, y_test=y_test_arr)
        twin_mask = twin_info["feature_twin_mask"]
        unseen_mask = ~twin_mask

        unseen_stats = _compute_subset_metrics(preds[unseen_mask], y_test_arr[unseen_mask], labels, benign_name)
        twins_only_stats = _compute_subset_metrics(preds[twin_mask], y_test_arr[twin_mask], labels, benign_name)

        per_class_twins = {}
        for c in labels:
            c_mask = (y_test_arr == c)
            c_total = int(c_mask.sum())
            c_twins = int((c_mask & twin_mask).sum())
            c_unseen = c_total - c_twins
            f1_all = full_stats["per_class"][c]["f1-score"]
            f1_unseen = unseen_stats["per_class"].get(c, {}).get("f1-score") if c_unseen > 0 else None
            per_class_twins[c] = {
                "test_support": c_total,
                "train_twins": c_twins,
                "unseen_support": c_unseen,
                "pct_twins": round(100.0 * c_twins / c_total, 2) if c_total > 0 else 0.0,
                "f1_all": f1_all,
                "f1_unseen": f1_unseen,
                "delta_f1_unseen_minus_all": round(f1_unseen - f1_all, 4) if f1_unseen is not None else None,
            }

        delta_f1 = (
            round(unseen_stats["macro_f1"] - full_stats["macro_f1"], 4)
            if unseen_stats["macro_f1"] is not None
            else None
        )
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
            "macro_f1_twins_only": twins_only_stats["macro_f1"],
            "macro_f1_unseen": unseen_stats["macro_f1"],
            "delta_macro_f1_unseen_minus_all": delta_f1,
            "benign_fpr_all": full_stats["benign_fpr"],
            "benign_fpr_twins_only": twins_only_stats["benign_fpr"],
            "benign_fpr_unseen": unseen_stats["benign_fpr"],
            "delta_benign_fpr_unseen_minus_all": delta_fpr,
            "per_class_breakdown": per_class_twins,
        }

    return result


def _json_safe(val):
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        return val if np.isfinite(val) else None
    if isinstance(val, np.generic):
        return _json_safe(val.item())
    if isinstance(val, np.ndarray):
        return [_json_safe(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    return repr(val)


def _model_params(model) -> dict:
    """Extracts JSON-safe hyperparameters from an estimator or Pipeline via get_params()."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(getattr(est, "hyperparams", None), dict) and est.hyperparams:
        raw = {**est.hyperparams, "seed": getattr(est, "seed", SEED)}
    else:
        raw = est.get_params(deep=False) if hasattr(est, "get_params") else {}
    return {str(k): _json_safe(v) for k, v in raw.items()}


def _resolve_seed(model, seed: int | None) -> int:
    """Returns an explicit seed if provided, else the model's random_state/seed, else SEED (42)."""
    if seed is not None:
        return int(seed)
    candidates = (
        [s for _, s in reversed(model.steps)] + [model]
        if hasattr(model, "steps")
        else [model]
    )
    for candidate in candidates:
        for attr in ("random_state", "seed"):
            val = getattr(candidate, attr, None)
            if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                return int(val)
    if hasattr(model, "get_params"):
        params = model.get_params()
        for key in ("random_state", "seed"):
            val = params.get(key)
            if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                return int(val)
        for key, val in reversed(list(params.items())):
            if key.endswith(("__random_state", "__seed")) and isinstance(
                val, (int, np.integer)
            ) and not isinstance(val, bool):
                return int(val)
    return SEED


def run_eval_template(
    model_or_pair,
    data_dir: Path = Path("data/processed"),
    task: str = "multiclass",
    model_name: str = "custom_model",
    out_dir: Path = Path("results"),
    use_wandb: bool = USE_WANDB,
    seed: int | None = None,
) -> list[dict]:
    """
    Runs the full evaluation suite with and without dst_port, including train-twin overlap,
    score changes when exact train feature twins are removed, and detailed W&B logging
    (parameters, model_params, seed, data_version, code_commit, and full metrics).
    """
    import time
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    data_meta = get_data_metadata(data_dir)
    commit_hash = get_code_commit()

    results = []
    for use_port in (True, False):
        X_tr, y_tr, _ = prep_xy(train_df, use_port, task)
        X_te, y_te, benign_name = prep_xy(test_df, use_port, task)

        t0 = time.time()
        # Handle pre-trained model pair vs. unfitted estimator blueprint
        if isinstance(model_or_pair, dict):
            model = model_or_pair["with_port" if use_port else "no_port"]
            fit_sec = 0.0
        else:
            model = clone(model_or_pair).fit(X_tr, y_tr)
            fit_sec = round(time.time() - t0, 2)

        resolved_seed = _resolve_seed(model, seed)
        params = _model_params(model)

        t_inf = time.time()
        metrics = evaluate_single(model, X_te, y_te, benign_name, X_train=X_tr, y_train=y_tr)
        infer_sec = round(time.time() - t_inf, 2)

        metrics.update({
            "model": model_name,
            "task": task,
            "dst_port": use_port,
            "n_features": X_te.shape[1],
            "seed": resolved_seed,
            "data_version": data_meta["data_version"],
            "code_commit": commit_hash,
            "model_params": params,
            "train_rows": len(X_tr),
            "test_rows": len(X_te),
            "fit_time_sec": fit_sec,
            "infer_time_sec": infer_sec,
        })
        results.append(metrics)

        tw = metrics["train_twin_analysis"]
        top_3 = ", ".join(f"{f['feature']} ({f['importance']:.3f})" for f in metrics["top_features"][:3])
        print(
            f"{model_name:12} | {task:10} | port={str(use_port):5} | "
            f"Twins={tw['n_feature_twins']:,}/{tw['n_test_rows']:,} ({tw['pct_feature_twins']:.1f}%) | "
            f"F1(all->unseen)={metrics['macro_f1']:.4f}->{tw['macro_f1_unseen']:.4f} "
            f"(Δ={tw['delta_macro_f1_unseen_minus_all']:+.4f}) | "
            f"FPR(all->unseen)={metrics['benign_fpr']:.4f}->{tw['benign_fpr_unseen']:.4f} | Top: {top_3}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{model_name}_{task}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved full report to {out_file}\n")

    if use_wandb:
        import wandb
        for r in results:
            tw = r["train_twin_analysis"]
            run_name = f"eval_{r['model']}_{r['task']}_{'port' if r['dst_port'] else 'noport'}"
            run_config = {
                **r["model_params"],
                **data_meta,
                "model_params": r["model_params"],
                "parameters": r["parameters"],
                "model": r["model"],
                "task": r["task"],
                "dst_port": r["dst_port"],
                "n_features": r["n_features"],
                "seed": r["seed"],
                "data_version": r["data_version"],
                "code_commit": r["code_commit"],
                "train_rows": r["train_rows"],
                "test_rows": r["test_rows"],
            }
            with wandb.init(**get_wandb_init_kwargs(job_type="eval", name=run_name, config=run_config)) as run:
                log_payload = {
                    "macro_f1": r["macro_f1"],
                    "weighted_f1": r["weighted_f1"],
                    "accuracy": r["accuracy"],
                    "benign_fpr": r["benign_fpr"],
                    "macro_f1_twins_only": tw["macro_f1_twins_only"],
                    "macro_f1_unseen": tw["macro_f1_unseen"],
                    "benign_fpr_unseen": tw["benign_fpr_unseen"],
                    "delta_macro_f1_unseen_minus_all": tw["delta_macro_f1_unseen_minus_all"],
                    "delta_benign_fpr_unseen_minus_all": tw["delta_benign_fpr_unseen_minus_all"],
                    "n_feature_twins": tw["n_feature_twins"],
                    "pct_feature_twins": tw["pct_feature_twins"],
                    "n_exact_twins": tw["n_exact_twins"],
                    "pct_exact_twins": tw["pct_exact_twins"],
                    "n_unseen_rows": tw["n_unseen_rows"],
                    "fit_time_sec": r["fit_time_sec"],
                    "infer_time_sec": r["infer_time_sec"],
                }
                for c_label, c_stats in r["per_class"].items():
                    log_payload[f"per_class_f1/{c_label}"] = c_stats["f1-score"]
                    log_payload[f"per_class_recall/{c_label}"] = c_stats["recall"]
                    log_payload[f"per_class_precision/{c_label}"] = c_stats["precision"]
                for c_label, c_tw in tw["per_class_breakdown"].items():
                    if c_tw["f1_unseen"] is not None:
                        log_payload[f"per_class_f1_unseen/{c_label}"] = c_tw["f1_unseen"]
                if r["top_features"] and hasattr(wandb, "Table"):
                    feat_table = wandb.Table(
                        columns=["rank", "feature", "importance"],
                        data=[[idx + 1, f["feature"], f["importance"]] for idx, f in enumerate(r["top_features"])],
                    )
                    log_payload["top_features_table"] = feat_table
                run.log(log_payload)

    return results



import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Identifier and metadata columns excluded to prevent data leakage (SEN-FR-03)
ID_COLS = [
    "id", "expiration_id", "flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms", "dst2src_first_seen_ms",
    "dst2src_last_seen_ms", "timestamp", "label"
]


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
        perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, scoring="f1_macro")
        scores = perm.importances_mean

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": X_test.columns[i], "importance": round(float(scores[i]), 5)} for i in top_idx]


def evaluate_single(model, X_test: pd.DataFrame, y_test, benign_name: str, top_k: int = 10) -> dict:
    """Computes all required metrics on a single fitted model and test split."""
    preds = model.predict(X_test)
    labels = sorted(pd.unique(y_test))

    report = classification_report(y_test, preds, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
        for c in labels
    }

    is_benign = (y_test == benign_name)
    benign_fpr = float((preds[is_benign] != benign_name).mean()) if is_benign.any() else None

    return {
        "macro_f1": round(float(f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)), 4),
        "benign_fpr": round(benign_fpr, 4) if benign_fpr is not None else None,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
        },
        "top_features": get_top_features(model, X_test, y_test, top_k=top_k),
    }


def run_eval_template(
    model_or_pair,
    data_dir: Path = Path("data/processed"),
    task: str = "multiclass",
    model_name: str = "custom_model",
    out_dir: Path = Path("results"),
    use_wandb: bool = False,
) -> list[dict]:
    """
    Runs the full evaluation suite with and without dst_port.
    - Pass an unfitted estimator to auto-train and test both variants.
    - Or pass a dict {"with_port": fitted_m1, "no_port": fitted_m2} to only run test inference.
    """
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    results = []
    for use_port in (True, False):
        X_tr, y_tr, _ = prep_xy(train_df, use_port, task)
        X_te, y_te, benign_name = prep_xy(test_df, use_port, task)

        # Handle pre-trained model pair vs. unfitted estimator blueprint
        if isinstance(model_or_pair, dict):
            model = model_or_pair["with_port" if use_port else "no_port"]
        else:
            model = clone(model_or_pair).fit(X_tr, y_tr)

        metrics = evaluate_single(model, X_te, y_te, benign_name)
        metrics.update({"model": model_name, "task": task, "dst_port": use_port, "n_features": X_te.shape[1]})
        results.append(metrics)

        top_3 = ", ".join(f"{f['feature']} ({f['importance']:.3f})" for f in metrics["top_features"][:3])
        print(f"{model_name:12} | {task:10} | port={str(use_port):5} | "
              f"Macro-F1={metrics['macro_f1']:.4f} | FPR={metrics['benign_fpr']:.4f} | Top: {top_3}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{model_name}_{task}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved full report to {out_file}\n")

    if use_wandb:
        import wandb
        for r in results:
            with wandb.init(
                entity="wcs-sentinel",
                project="Sentinel",
                job_type="eval",
                config={k: r[k] for k in ("model", "task", "dst_port", "n_features")},
                reinit="finish_previous",
            ) as run:
                run.log({"macro_f1": r["macro_f1"], "benign_fpr": r["benign_fpr"]})

    return results

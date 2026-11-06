#FR07
"""SEN-FR-07: Sanity baselines (majority class + logistic regression) and leakage check."""
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
EVAL_SPLIT = "val"  # Change to "test" for final evaluation
USE_WANDB = False   # Change to True to log to Weights & Biases (SEN-FR-10)

# Identifier and metadata columns excluded to prevent data leakage (SEN-FR-03)
ID_COLS = [
    "id", "expiration_id", "flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms", "dst2src_first_seen_ms",
    "dst2src_last_seen_ms", "timestamp", "label"
]


def run_baselines(data_dir=DATA_DIR, eval_split=EVAL_SPLIT, use_wandb=USE_WANDB, out_dir=Path("results"), train_sample=None):
    import time
    print(f"Loading train split...", flush=True)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    if train_sample is not None:
        train_df = train_df.sample(n=min(train_sample, len(train_df)), random_state=42)
    print(f"  train rows: {len(train_df):,}", flush=True)
    print(f"Loading {eval_split} split...", flush=True)
    eval_df = pd.read_parquet(data_dir / f"{eval_split}.parquet")
    print(f"  {eval_split} rows: {len(eval_df):,}", flush=True)

    combos = list(product(["binary", "multiclass"], [True, False]))
    results = []
    # Evaluate all combinations of task (binary vs. multiclass) and destination port inclusion
    for combo_i, (task, use_port) in enumerate(combos, 1):
        drop_cols = ID_COLS + ([] if use_port else ["dst_port"])
        X_tr = train_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
        X_ev = eval_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")

        # Collapse attack classes into a single "attack" label for binary classification
        y_tr = np.where(train_df["label"] == "BENIGN", "benign", "attack") if task == "binary" else train_df["label"]
        y_ev = np.where(eval_df["label"] == "BENIGN", "benign", "attack") if task == "binary" else eval_df["label"]
        benign_label = "benign" if task == "binary" else "BENIGN"

        models = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="saga", verbose=1)),
        }

        for name, model in models.items():
            print(f"[{combo_i}/{len(combos)}] Fitting {name} | task={task} | port={use_port} | {X_tr.shape[0]:,} rows x {X_tr.shape[1]} features...", flush=True)
            t0 = time.time()
            preds = model.fit(X_tr, y_tr).predict(X_ev)
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s", flush=True)
            labels = sorted(pd.unique(y_ev))
            f1 = float(f1_score(y_ev, preds, average="macro", zero_division=0))
            # False positive rate: fraction of true benign flows misclassified as an attack
            fpr = float((preds[y_ev == benign_label] != benign_label).mean())

            results.append({
                "model": name, "task": task, "dst_port": use_port, "macro_f1": f1, "benign_fpr": fpr,
                "per_class": classification_report(y_ev, preds, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_ev, preds, labels=labels).tolist(),
            })
            # Near-perfect baseline F1 indicates target or split leakage (Risk R2)
            alarm = " <-- LEAKAGE ALARM" if f1 >= 0.999 else ""
            print(f"{task:10} | {name:8} | port={str(use_port):5} | F1={f1:.4f} | FPR={fpr:.4f}{alarm}")

    out_dir.mkdir(exist_ok=True)
    (out_dir / f"baselines_{eval_split}.json").write_text(json.dumps(results, indent=2))

    # Optional W&B logging for each baseline run (SEN-FR-10)
    if use_wandb:
        import wandb
        for r in results:
            with wandb.init(entity="wcs-sentinel", project="Sentinel", job_type="baseline", config={k: r[k] for k in ("model", "task", "dst_port")}, reinit="finish_previous") as run:
                run.log({"macro_f1": r["macro_f1"], "benign_fpr": r["benign_fpr"]})

    return results


if __name__ == "__main__":
    run_baselines()

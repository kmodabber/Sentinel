#FR07
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
ID_COLS = [
    "id", "expiration_id", "flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms", "dst2src_first_seen_ms",
    "dst2src_last_seen_ms", "timestamp", "label"
]


def run_baselines(data_dir=DATA_DIR, eval_split=EVAL_SPLIT, use_wandb=USE_WANDB, out_dir=Path("results")):
    train_df = pd.read_parquet(data_dir / "train.parquet")
    eval_df = pd.read_parquet(data_dir / f"{eval_split}.parquet")

    results = []
    for task, use_port in product(["binary", "multiclass"], [True, False]):
        drop_cols = ID_COLS + ([] if use_port else ["dst_port"])
        X_tr = train_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
        X_ev = eval_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")

        y_tr = np.where(train_df["label"] == "BENIGN", "benign", "attack") if task == "binary" else train_df["label"]
        y_ev = np.where(eval_df["label"] == "BENIGN", "benign", "attack") if task == "binary" else eval_df["label"]
        benign_label = "benign" if task == "binary" else "BENIGN"

        models = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        }

        for name, model in models.items():
            preds = model.fit(X_tr, y_tr).predict(X_ev)
            labels = sorted(pd.unique(y_ev))
            f1 = float(f1_score(y_ev, preds, average="macro", zero_division=0))
            fpr = float((preds[y_ev == benign_label] != benign_label).mean())

            results.append({
                "model": name, "task": task, "dst_port": use_port, "macro_f1": f1, "benign_fpr": fpr,
                "per_class": classification_report(y_ev, preds, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_ev, preds, labels=labels).tolist(),
            })
            alarm = " <-- LEAKAGE ALARM" if f1 >= 0.999 else ""
            print(f"{task:10} | {name:8} | port={str(use_port):5} | F1={f1:.4f} | FPR={fpr:.4f}{alarm}")

    out_dir.mkdir(exist_ok=True)
    (out_dir / f"baselines_{eval_split}.json").write_text(json.dumps(results, indent=2))

    if use_wandb:
        import wandb
        for r in results:
            with wandb.init(project="sentinel", job_type="baseline", config={k: r[k] for k in ("model", "task", "dst_port")}, reinit=True) as run:
                run.log({"macro_f1": r["macro_f1"], "benign_fpr": r["benign_fpr"]})

    return results


if __name__ == "__main__":
    run_baselines()

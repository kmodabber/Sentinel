"""SEN-FR-13: train a small PyTorch MLP on the NFS-2023-TE splits and feed it
through the shared evaluation template so results line up with the trees and
the logistic-regression baseline.

Run from the repo root (after sentinel_data.split has produced the Parquet
files):

    python -m sentinel_ml.train_mlp --task both
    python -m sentinel_ml.train_mlp --task multiclass --train-sample 500000 --use-wandb

Produces, per (task, use_port):
    models/preprocess_mlp_{task}_{port}.joblib   (fitted on TRAIN only)
    models/mlp_{task}_{port}.pt                  (best epoch by val macro-F1)
    results/train_mlp_{task}_{port}.json         (hyper-params, val history, meta)
    results/eval_mlp_{task}.json                 (written by eval_template)

Val is used for early stopping and model selection. Test is evaluated once at
the end through sentinel_ml.eval_template.run_eval_template so metrics are
directly comparable with the other models.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from sentinel_ml.eval_template import prep_xy, run_eval_template

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

DEFAULT_HIDDEN = (128, 64)
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 1024
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42

LEAKAGE_F1_THRESHOLD = 0.999
LEAKAGE_LOGREG_MARGIN = 0.01


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # some ops have no deterministic kernel on CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def data_version(data_dir: Path) -> str | None:
    card = data_dir / "data_card.json"
    if not card.exists():
        return None
    try:
        return json.loads(card.read_text()).get("version")
    except (OSError, json.JSONDecodeError):
        return None


def build_preprocessor(n_train_rows: int) -> Pipeline:
    """QuantileTransformer handles the negative values NFStream can emit, then
    StandardScaler keeps the inputs neural-friendly. Fitted on train only."""
    n_quantiles = int(min(1000, max(10, n_train_rows)))
    return Pipeline([
        ("quantile", QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=200_000,
            random_state=DEFAULT_SEED,
        )),
        ("scaler", StandardScaler()),
    ])


def class_weights_from_train(y_train_encoded: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weights computed on the training labels only."""
    classes = np.arange(n_classes)
    return compute_class_weight("balanced", classes=classes, y=y_train_encoded)


class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SklearnMLPWrapper(ClassifierMixin, BaseEstimator):
    """Adapts a trained torch MLP to the predict(DataFrame) contract used by
    sentinel_ml.eval_template.evaluate_single. Inherits from BaseEstimator so
    permutation_importance (and anything else sklearn-side) is happy."""

    def __init__(self, model: MLP, preprocessor: Pipeline, label_encoder: LabelEncoder, feature_names: list[str]):
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.feature_names = list(feature_names)
        self.classes_ = np.array(label_encoder.classes_)

    def fit(self, X=None, y=None):  # no-op; model is already trained
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Reorder to the training columns so a rearranged test frame still works.
        X = X[self.feature_names]
        Xt = self.preprocessor.transform(X.to_numpy(dtype=np.float32))
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xt).float())
            idx = logits.argmax(dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(idx)


@dataclass
class TrainOutcome:
    best_epoch: int
    best_val_macro_f1: float
    epochs_trained: int
    train_time_s: float
    val_history: list[dict] = field(default_factory=list)


def _macro_f1_from_logits(logits: torch.Tensor, y_true: np.ndarray, n_classes: int) -> float:
    """Light macro-F1 that avoids an sklearn round-trip inside the training loop."""
    preds = logits.argmax(dim=1).cpu().numpy()
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (y_true == c)).sum())
        fp = int(((preds == c) & (y_true != c)).sum())
        fn = int(((preds != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def train_one(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int,
    class_weights: np.ndarray,
    *, hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT,
    lr=DEFAULT_LR, batch_size=DEFAULT_BATCH,
    epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, seed=DEFAULT_SEED,
) -> tuple[MLP, TrainOutcome]:
    set_seed(seed)
    model = MLP(X_tr.shape[1], n_classes, hidden=hidden, dropout=dropout)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).long(),
    )
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)

    X_val_t = torch.from_numpy(X_val).float()

    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    bad = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val).long()).item())
        val_f1 = _macro_f1_from_logits(val_logits, y_val, n_classes)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}", flush=True)

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    outcome = TrainOutcome(
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        epochs_trained=len(history),
        train_time_s=time.time() - t0,
        val_history=history,
    )
    return model, outcome


def _maybe_subsample(X: pd.DataFrame, y: np.ndarray, n: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray, bool]:
    if n is None or n >= len(X):
        return X, y, False
    # Stratified; fall back to a plain sample if some class is too small.
    try:
        X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
        return X_s.reset_index(drop=True), np.asarray(y_s), True
    except ValueError:
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        return X.iloc[idx].reset_index(drop=True), y[idx], True


def _leakage_warning(task: str, use_port: bool, val_f1: float, test_f1: float, logreg_test_f1: float | None) -> list[str]:
    msgs: list[str] = []
    label = f"{task} / port={use_port}"
    if val_f1 >= LEAKAGE_F1_THRESHOLD:
        msgs.append(f"LEAKAGE ALARM [{label}]: val macro-F1 {val_f1:.4f} >= {LEAKAGE_F1_THRESHOLD}")
    if test_f1 >= LEAKAGE_F1_THRESHOLD:
        msgs.append(f"LEAKAGE ALARM [{label}]: test macro-F1 {test_f1:.4f} >= {LEAKAGE_F1_THRESHOLD}")
    if logreg_test_f1 is not None and abs(test_f1 - logreg_test_f1) <= LEAKAGE_LOGREG_MARGIN:
        msgs.append(
            f"LEAKAGE ALARM [{label}]: MLP macro-F1 {test_f1:.4f} is within "
            f"{LEAKAGE_LOGREG_MARGIN} of logistic regression {logreg_test_f1:.4f} - "
            f"the MLP may be learning nothing extra, or a leaky feature is dominating."
        )
    return msgs


def _logreg_test_f1(results_dir: Path, task: str, use_port: bool) -> float | None:
    """Look up the logistic-regression test macro-F1 for the leakage comparison.
    Checks both the FR-07 baselines output and an eval-template logreg report."""
    bfile = results_dir / "baselines_test.json"
    if bfile.exists():
        try:
            for r in json.loads(bfile.read_text()):
                if r.get("model") == "logreg" and r.get("task") == task and bool(r.get("dst_port")) == use_port:
                    return float(r.get("macro_f1"))
        except (OSError, json.JSONDecodeError):
            pass
    efile = results_dir / f"eval_logreg_{task}.json"
    if efile.exists():
        try:
            for r in json.loads(efile.read_text()):
                if bool(r.get("dst_port")) == use_port:
                    return float(r.get("macro_f1"))
        except (OSError, json.JSONDecodeError):
            pass
    return None


def train_task(
    task: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *, data_dir: Path, models_dir: Path, results_dir: Path,
    use_port_variants: tuple[bool, ...] = (True, False),
    train_sample: int | None = None,
    hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT,
    lr=DEFAULT_LR, batch_size=DEFAULT_BATCH,
    epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, seed=DEFAULT_SEED,
    use_wandb: bool = False,
) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    version = data_version(data_dir)

    per_variant: dict[str, SklearnMLPWrapper] = {}
    meta_by_variant: dict[str, dict] = {}

    for use_port in use_port_variants:
        tag = "with_port" if use_port else "no_port"
        print(f"\n[{task}] fitting MLP, dst_port={'on' if use_port else 'off'}", flush=True)

        X_tr_df, y_tr_raw, _ = prep_xy(train_df, use_port=use_port, task=task)
        X_val_df, y_val_raw, _ = prep_xy(val_df, use_port=use_port, task=task)

        X_tr_df, y_tr_raw, subsampled = _maybe_subsample(X_tr_df, y_tr_raw, train_sample, seed)
        if subsampled:
            print(f"    stratified sub-sample of TRAIN to {len(X_tr_df):,} rows (val/test untouched)", flush=True)

        le = LabelEncoder().fit(y_tr_raw)
        # Val labels unseen at train time (shouldn't happen after FR-05, but guard anyway).
        known = np.isin(y_val_raw, le.classes_)
        if not known.all():
            dropped = int((~known).sum())
            print(f"    dropping {dropped} val rows with labels unseen in train", flush=True)
            X_val_df = X_val_df.loc[known].reset_index(drop=True)
            y_val_raw = y_val_raw[known]
        y_tr = le.transform(y_tr_raw).astype(np.int64)
        y_val = le.transform(y_val_raw).astype(np.int64)
        n_classes = len(le.classes_)

        pre = build_preprocessor(len(X_tr_df)).fit(X_tr_df.to_numpy(dtype=np.float32))
        X_tr = pre.transform(X_tr_df.to_numpy(dtype=np.float32)).astype(np.float32)
        X_val = pre.transform(X_val_df.to_numpy(dtype=np.float32)).astype(np.float32)

        cw = class_weights_from_train(y_tr, n_classes)
        print(f"    features={X_tr.shape[1]}  classes={n_classes}  class_weights=[{', '.join(f'{w:.2f}' for w in cw)}]", flush=True)

        model, outcome = train_one(
            X_tr, y_tr, X_val, y_val, n_classes, cw,
            hidden=hidden, dropout=dropout, lr=lr, batch_size=batch_size,
            epochs=epochs, patience=patience, seed=seed,
        )
        print(f"    best epoch {outcome.best_epoch} val macro-F1 {outcome.best_val_macro_f1:.4f} "
              f"({outcome.epochs_trained} epochs in {outcome.train_time_s:.1f}s)", flush=True)

        pre_path = models_dir / f"preprocess_mlp_{task}_{tag}.joblib"
        model_path = models_dir / f"mlp_{task}_{tag}.pt"
        joblib.dump({"preprocessor": pre, "label_encoder": le, "feature_names": list(X_tr_df.columns)}, pre_path)
        torch.save({
            "state_dict": model.state_dict(),
            "in_features": X_tr.shape[1],
            "n_classes": n_classes,
            "hidden": list(hidden),
            "dropout": dropout,
        }, model_path)

        wrapper = SklearnMLPWrapper(model, pre, le, list(X_tr_df.columns))
        per_variant[tag] = wrapper
        meta_by_variant[tag] = {
            "best_epoch": outcome.best_epoch,
            "best_val_macro_f1": outcome.best_val_macro_f1,
            "epochs_trained": outcome.epochs_trained,
            "train_time_s": round(outcome.train_time_s, 2),
            "n_features": int(X_tr.shape[1]),
            "n_train_rows": int(len(X_tr_df)),
            "n_val_rows": int(len(X_val_df)),
            "classes": [str(c) for c in le.classes_],
            "class_weights": [float(w) for w in cw],
            "val_history": outcome.val_history,
            "artefacts": {"preprocessor": str(pre_path), "model": str(model_path)},
        }

    # Final test evaluation through the shared template (same metrics as trees/logreg).
    test_results = run_eval_template(
        per_variant,
        data_dir=data_dir,
        task=task,
        model_name="mlp",
        out_dir=results_dir,
        use_wandb=False,  # W&B handled below so we can log train hyper-params too.
    )

    alarms: list[str] = []
    for r in test_results:
        tag = "with_port" if r["dst_port"] else "no_port"
        val_f1 = meta_by_variant[tag]["best_val_macro_f1"]
        logreg_f1 = _logreg_test_f1(results_dir, task, r["dst_port"])
        alarms.extend(_leakage_warning(task, r["dst_port"], val_f1, r["macro_f1"], logreg_f1))
    if alarms:
        print("\n".join("  " + a for a in alarms), flush=True)

    train_meta = {
        "task": task,
        "data_version": version,
        "code_commit": commit,
        "seed": seed,
        "hyperparams": {
            "hidden": list(hidden), "dropout": dropout, "lr": lr,
            "batch_size": batch_size, "epochs": epochs, "patience": patience,
        },
        "train_sample": train_sample,
        "variants": meta_by_variant,
        "leakage_alarms": alarms,
    }
    for use_port in use_port_variants:
        tag = "with_port" if use_port else "no_port"
        out = results_dir / f"train_mlp_{task}_{tag}.json"
        out.write_text(json.dumps({**train_meta, "variant": tag}, indent=2))

    if use_wandb:
        import wandb
        for r in test_results:
            tag = "with_port" if r["dst_port"] else "no_port"
            vmeta = meta_by_variant[tag]
            with wandb.init(
                entity="wcs-sentinel", project="Sentinel", job_type="train",
                config={
                    "model": "mlp", "task": task, "dst_port": r["dst_port"],
                    "data_version": version, "code_commit": commit,
                    "seed": seed, "train_sample": train_sample,
                    **{f"hp_{k}": v for k, v in train_meta["hyperparams"].items()},
                    "n_features": vmeta["n_features"],
                    "n_train_rows": vmeta["n_train_rows"],
                },
                reinit="finish_previous",
            ) as run:
                run.log({
                    "val_macro_f1": vmeta["best_val_macro_f1"],
                    "test_macro_f1": r["macro_f1"],
                    "test_benign_fpr": r["benign_fpr"],
                    "best_epoch": vmeta["best_epoch"],
                    "epochs_trained": vmeta["epochs_trained"],
                    "leakage_alarms": len(alarms),
                })

    return {"task": task, "test_results": test_results, "meta": train_meta}


def main(
    *, task: str = "both",
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    train_sample: int | None = None,
    hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT,
    lr=DEFAULT_LR, batch_size=DEFAULT_BATCH,
    epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, seed=DEFAULT_SEED,
    use_wandb: bool = False,
    use_port_variants: tuple[bool, ...] = (True, False),
) -> list[dict]:
    tasks = ["binary", "multiclass"] if task == "both" else [task]
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    print(f"train={len(train_df):,} rows   val={len(val_df):,} rows", flush=True)

    out = []
    for t in tasks:
        out.append(train_task(
            t, train_df, val_df,
            data_dir=data_dir, models_dir=models_dir, results_dir=results_dir,
            use_port_variants=use_port_variants,
            train_sample=train_sample,
            hidden=hidden, dropout=dropout, lr=lr, batch_size=batch_size,
            epochs=epochs, patience=patience, seed=seed,
            use_wandb=use_wandb,
        ))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEN-FR-13 MLP trainer")
    p.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--train-sample", type=int, default=None,
                   help="Stratified sub-sample of TRAIN only (val/test are always full).")
    p.add_argument("--hidden", type=int, nargs=2, default=list(DEFAULT_HIDDEN))
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        task=args.task, data_dir=args.data_dir, models_dir=args.models_dir,
        results_dir=args.results_dir, train_sample=args.train_sample,
        hidden=tuple(args.hidden), dropout=args.dropout, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        seed=args.seed, use_wandb=args.use_wandb,
    )

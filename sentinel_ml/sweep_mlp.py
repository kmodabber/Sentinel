"""SEN-FR-13: light validation-only sweep for the MLP.

Runs a small grid over (hidden, lr, weight_decay, dropout). Picks the best
config by val macro-F1. Test is NEVER touched. Writes results to
``results/sweep_mlp_{task}_{tag}.json`` so decisions are auditable.

Usage (from repo root, after sentinel_data.split has produced the splits):

    python -m sentinel_ml.sweep_mlp --task multiclass --train-sample 200000
    python -m sentinel_ml.sweep_mlp --task binary --use-port both
"""
from __future__ import annotations

# isort: off
# torch first to avoid Windows OpenMP clash with numpy.
import torch  # noqa: F401,E402

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sentinel_features.features import LABELS as CANONICAL_LABELS
from sentinel_ml.baselines import _rare_from_canonical, canonicalize_labels
from sentinel_ml.eval_template import prep_xy
from sentinel_ml.train_mlp import (
    _maybe_subsample,
    build_preprocessor,
    class_weights_from_train,
    task_label_set,
)
# isort: on

DEFAULT_GRID = {
    "hidden": [(64, 32), (128, 64), (256, 128)],
    "lr": [5e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4],
    "dropout": [0.1, 0.3],
}


def apply_weight_decay(model_params, lr: float, weight_decay: float):
    return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)


def _fit_and_score(X_tr, y_tr, X_val, y_val, n_classes, cw, *, hidden, dropout, lr, weight_decay, epochs, patience, seed):
    """Wrap train_one with a weight-decay-aware optimizer. train_one creates its
    own Adam without wd, so we monkey-patch via torch.optim.Adam default.
    """
    # Easiest path: call train_one and let it build Adam(lr=lr). To add wd, we
    # re-implement the loop here in the smallest way possible.
    import torch.nn as nn  # noqa: I001
    from torch.utils.data import DataLoader, TensorDataset
    from sentinel_ml.train_mlp import (
        DEFAULT_BATCH,
        MLP,
        _macro_f1_from_logits,
        set_seed,
    )

    set_seed(seed)
    model = MLP(X_tr.shape[1], n_classes, hidden=hidden, dropout=dropout)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=DEFAULT_BATCH, shuffle=True, generator=g)
    X_val_t = torch.from_numpy(X_val).float()
    best_f1, best_epoch, bad = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
        f1 = _macro_f1_from_logits(val_logits, y_val, n_classes)
        if f1 > best_f1 + 1e-6:
            best_f1, best_epoch, bad = f1, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_f1, best_epoch


def sweep(task: str, use_port: bool, train_df: pd.DataFrame, val_df: pd.DataFrame,
          *, train_sample: int | None, grid: dict, epochs: int, patience: int,
          seed: int, out_path: Path) -> dict:
    X_tr_df, y_tr_raw, _ = prep_xy(train_df, use_port=use_port, task=task)
    X_val_df, y_val_raw, _ = prep_xy(val_df, use_port=use_port, task=task)
    X_tr_df, y_tr_raw, _ = _maybe_subsample(X_tr_df, y_tr_raw, train_sample, seed)
    label_set = task_label_set(task)
    le = LabelEncoder().fit(np.array(label_set))
    y_tr = le.transform(y_tr_raw).astype(np.int64)
    y_val = le.transform(y_val_raw).astype(np.int64)
    n_classes = len(le.classes_)

    pre = build_preprocessor(len(X_tr_df)).fit(X_tr_df.to_numpy(dtype=np.float32))
    X_tr = pre.transform(X_tr_df.to_numpy(dtype=np.float32)).astype(np.float32)
    X_val = pre.transform(X_val_df.to_numpy(dtype=np.float32)).astype(np.float32)
    cw = class_weights_from_train(y_tr, n_classes)

    combos = list(itertools.product(grid["hidden"], grid["lr"], grid["weight_decay"], grid["dropout"]))
    print(f"\n[sweep] task={task} port={use_port}  features={X_tr.shape[1]}  n_train={len(X_tr):,}  configs={len(combos)}", flush=True)

    trials = []
    t0 = time.time()
    for i, (hidden, lr, wd, dropout) in enumerate(combos, 1):
        ts = time.time()
        f1, best_epoch = _fit_and_score(
            X_tr, y_tr, X_val, y_val, n_classes, cw,
            hidden=hidden, dropout=dropout, lr=lr, weight_decay=wd,
            epochs=epochs, patience=patience, seed=seed,
        )
        dt = time.time() - ts
        print(f"  [{i:02d}/{len(combos)}] hidden={hidden} lr={lr} wd={wd} dropout={dropout} "
              f"-> val_macroF1={f1:.4f} best_epoch={best_epoch} ({dt:.1f}s)", flush=True)
        trials.append({
            "hidden": list(hidden), "lr": lr, "weight_decay": wd, "dropout": dropout,
            "val_macro_f1": round(f1, 4), "best_epoch": best_epoch, "time_s": round(dt, 2),
        })

    trials_sorted = sorted(trials, key=lambda r: r["val_macro_f1"], reverse=True)
    best = trials_sorted[0]
    print(f"[sweep] best: {best}  (total {time.time()-t0:.1f}s)", flush=True)

    out = {
        "task": task, "dst_port": use_port, "seed": seed,
        "train_sample": train_sample, "n_train_rows": int(len(X_tr)),
        "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in grid.items()},
        "epochs": epochs, "patience": patience,
        "trials": trials_sorted,
        "best": best,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[sweep] wrote {out_path}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--use-port", choices=["both", "with", "without"], default="both")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--train-sample", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_df = pd.read_parquet(args.data_dir / "train.parquet")
    val_df = pd.read_parquet(args.data_dir / "val.parquet")
    rare = _rare_from_canonical(train_df, CANONICAL_LABELS)
    train_df = canonicalize_labels(train_df, rare_class_list=rare)
    val_df = canonicalize_labels(val_df, rare_class_list=rare)

    port_variants = [True, False] if args.use_port == "both" else ([True] if args.use_port == "with" else [False])
    for use_port in port_variants:
        tag = "with_port" if use_port else "no_port"
        out_path = args.results_dir / f"sweep_mlp_{args.task}_{tag}.json"
        sweep(args.task, use_port, train_df, val_df,
              train_sample=args.train_sample, grid=DEFAULT_GRID,
              epochs=args.epochs, patience=args.patience, seed=args.seed,
              out_path=out_path)


if __name__ == "__main__":
    main()

"""SEN-109: Sanity baselines (majority class + logistic regression) on data v1.

Metrics are computed via the shared eval template (sentinel_ml.eval_template) so
the output matches what SEN-108 / eval report v1 consumes.
"""

import json
import subprocess
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel_data.split import RARE_LABEL, merge_rare_classes
from sentinel_features.features import LABELS as CANONICAL_LABELS
from sentinel_ml.eval_template import ID_COLS, evaluate_single, prep_xy  # noqa: F401

DATA_DIR = Path("data/processed")
EVAL_SPLIT = "test"
SEED = 42
WANDB_ENTITY = "wcs-sentinel"
WANDB_PROJECT = "baseline_v1"

LEAK_LOGREG_THRESHOLD = 0.95
LEAK_ANY_THRESHOLD = 0.999


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _data_metadata(data_dir: Path, rare_class_list: list[str] | None = None) -> dict:
    card = data_dir / "data_card.json"
    if not card.exists():
        return {"version": "unknown", "rare_class_policy": "unknown"}
    d = json.loads(card.read_text())
    rare = d.get("rare_classes", {}).get("merged") or rare_class_list or []
    return {
        "version": d.get("version", "unknown"),
        "rare_class_policy": (
            f"raw labels not in sentinel_features.LABELS are merged into "
            f"'{RARE_LABEL}' (SEN-FR-05)"
        ),
        "rare_classes_merged": sorted(rare),
    }


def _rare_from_canonical(df: pd.DataFrame, canonical_labels: list[str]) -> list[str]:
    """Raw labels that are present in df but not in the canonical class set.

    These are the labels that must be relabelled as ``RARE_LABEL`` for the
    output to match ``canonical_labels``. Using the canonical list as the
    source of truth sidesteps the min_flows threshold mismatch between
    ``split.RARE_MIN_FLOWS`` and what ``sentinel_features.LABELS`` actually
    kept separate (e.g. 'Web Attack - Brute Force' has 88 train flows but is
    a canonical class).
    """
    present = set(df["label"].astype(str).unique())
    return sorted(present - set(canonical_labels) - {RARE_LABEL})


def canonicalize_labels(
    df: pd.DataFrame,
    rare_class_list: list[str] | None = None,
    canonical_labels: list[str] = CANONICAL_LABELS,
) -> pd.DataFrame:
    """Apply the SEN-FR-05 rare-class merge, then assert the result matches the
    canonical class set from ``sentinel_features.features.LABELS``.

    If ``rare_class_list`` is ``None``, any raw label in ``df`` that is not in
    ``canonical_labels`` is merged into ``RARE_LABEL``. Pass an explicit list
    to override (e.g. the data card's ``rare_classes.merged`` field, or an
    empty list to disable the merge for a test).
    """
    if rare_class_list is None:
        rare_class_list = _rare_from_canonical(df, canonical_labels)
    out = merge_rare_classes(df, rare_class_list) if rare_class_list else df.copy()
    actual = set(out["label"].astype(str).unique())
    expected = set(canonical_labels)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra or missing:
        raise ValueError(
            f"Multi-class label set does not match canonical "
            f"({len(canonical_labels)} classes). "
            f"Extra: {extra or 'none'}. Missing: {missing or 'none'}."
        )
    return out


def leakage_flags(results: list[dict]) -> list[str]:
    """Flag any result that crosses the SEN-109 leakage thresholds.

    Any model with macro_f1 >= 0.999, or logreg with macro_f1 >= 0.95, is suspicious.
    This only reports; investigation is SEN-106.
    """
    flags = []
    for r in results:
        f1 = r["macro_f1"]
        tag = f"{r['model']}/{r['task']}/port={r['dst_port']}"
        if f1 >= LEAK_ANY_THRESHOLD:
            flags.append(f"CRITICAL {tag}: macro_f1={f1:.4f} >= {LEAK_ANY_THRESHOLD}")
        elif r["model"] == "logreg" and f1 >= LEAK_LOGREG_THRESHOLD:
            flags.append(
                f"WARNING  {tag}: logreg macro_f1={f1:.4f} >= {LEAK_LOGREG_THRESHOLD}"
            )
    return flags


def _build_models():
    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
                solver="lbfgs",
            ),
        ),
    }


def _write_markdown(
    out_path: Path, metadata: dict, results: list[dict], flags: list[str]
) -> None:
    lines = [
        f"# SEN-109 Sanity Baselines — data {metadata['data']['version']} (test split)",
        "",
        f"- Git commit: `{metadata['git_commit'][:12]}`",
        f"- Seed: {metadata['seed']}",
        f"- Rare-class policy: {metadata['data']['rare_class_policy']}",
        f"- Evaluated split: **{metadata['split_evaluated']}**",
        "",
        "| model | task | dst_port | macro_f1 | benign_fpr | n_features |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['task']} | {r['dst_port']} | "
            f"{r['macro_f1']:.4f} | {r['benign_fpr']:.4f} | {r['n_features']} |"
        )
    lines.append("")
    if flags:
        lines.append("## Leakage flags")
        lines.append("")
        lines.extend(f"- {f}" for f in flags)
    else:
        lines.append("_No leakage thresholds crossed._")
    lines.append("")
    out_path.write_text("\n".join(lines))


def run_baselines(
    data_dir: Path = DATA_DIR,
    eval_split: str = EVAL_SPLIT,
    use_wandb: bool = False,
    out_dir: Path = Path("results"),
    train_sample: int | None = None,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    drop_duplicates: bool = False,
    canonical_labels: list[str] | None = None,
) -> list[dict]:
    """Run majority + logreg baselines across binary/multiclass and with/without dst_port.

    Metric computation uses sentinel_ml.eval_template.evaluate_single so results match
    the team template format consumed by SEN-108.
    """
    train_df = pd.read_parquet(data_dir / "train.parquet")
    if train_sample is not None:
        train_df = train_df.sample(
            n=min(train_sample, len(train_df)), random_state=SEED
        )
    eval_df = pd.read_parquet(data_dir / f"{eval_split}.parquet")

    # SEN-FR-05: enforce the canonical 12-class set from sentinel_features.LABELS
    # by relabelling any raw label not in that set as 'Rare attack', then
    # asserting train/eval labels match the canonical set exactly.
    labels = canonical_labels if canonical_labels is not None else CANONICAL_LABELS
    rare_from_train = _rare_from_canonical(train_df, labels)
    train_df = canonicalize_labels(
        train_df, rare_class_list=rare_from_train, canonical_labels=labels
    )
    eval_df = canonicalize_labels(
        eval_df, rare_class_list=rare_from_train, canonical_labels=labels
    )
    print(
        f"canonicalized labels: {len(labels)} classes, merged rare -> "
        f"'{RARE_LABEL}': {rare_from_train or 'none'}",
        flush=True,
    )

    dedup_stats: dict[str, dict[str, int]] = {}
    if drop_duplicates:
        tr_before, ev_before = len(train_df), len(eval_df)
        train_df = train_df.drop_duplicates(keep="first").reset_index(drop=True)
        eval_df = eval_df.drop_duplicates(keep="first").reset_index(drop=True)
        dedup_stats = {
            "train": {
                "before": tr_before,
                "after": len(train_df),
                "dropped": tr_before - len(train_df),
            },
            eval_split: {
                "before": ev_before,
                "after": len(eval_df),
                "dropped": ev_before - len(eval_df),
            },
        }
        print(
            f"dedup: train {tr_before:,} -> {len(train_df):,} | "
            f"{eval_split} {ev_before:,} -> {len(eval_df):,}",
            flush=True,
        )
    print(
        f"train rows: {len(train_df):,} | {eval_split} rows: {len(eval_df):,}",
        flush=True,
    )

    model_factories = _build_models()
    combos = list(product(["binary", "multiclass"], [True, False]))

    results = []
    for combo_i, (task, use_port) in enumerate(combos, 1):
        X_tr, y_tr, _ = prep_xy(train_df, use_port, task)
        X_ev, y_ev, benign_name = prep_xy(eval_df, use_port, task)

        for name, make_model in model_factories.items():
            model = make_model()
            print(
                f"[{combo_i}/{len(combos)}] {name} | task={task} | port={use_port} | "
                f"{X_tr.shape[0]:,}x{X_tr.shape[1]} ...",
                flush=True,
            )
            t0 = time.time()
            model.fit(X_tr, y_tr)
            if name == "majority":
                # DummyClassifier has no coef_/feature_importances_, so evaluate_single
                # would fall back to a very slow permutation_importance pass. Feature
                # importance for a constant predictor is meaningless; short-circuit it.
                model.feature_importances_ = np.zeros(X_ev.shape[1])
            metrics = evaluate_single(model, X_ev, y_ev, benign_name)
            metrics.update(
                {
                    "model": name,
                    "task": task,
                    "dst_port": use_port,
                    "n_features": X_ev.shape[1],
                }
            )
            print(
                f"  done in {time.time() - t0:.1f}s | "
                f"F1={metrics['macro_f1']:.4f} | FPR={metrics['benign_fpr']:.4f}",
                flush=True,
            )
            results.append(metrics)

    flags = leakage_flags(results)
    metadata = {
        "sen_ticket": "SEN-109",
        "seed": SEED,
        "git_commit": _git_commit(),
        "data": _data_metadata(data_dir, rare_class_list=rare_from_train),
        "split_evaluated": eval_split,
        "drop_duplicates": drop_duplicates,
        "dedup_stats": dedup_stats,
        "n_classes_multiclass": len(labels),
        "canonical_labels": list(labels),
        "dst_port_settings": [True, False],
        "leakage_thresholds": {
            "logreg_macro_f1": LEAK_LOGREG_THRESHOLD,
            "any_macro_f1": LEAK_ANY_THRESHOLD,
        },
        "leakage_flags": flags,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dedup" if drop_duplicates else ""
    (out_dir / f"baselines_{eval_split}{suffix}.json").write_text(
        json.dumps(results, indent=2)
    )
    (out_dir / f"baselines_{eval_split}{suffix}_meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    _write_markdown(
        out_dir / f"baselines_{eval_split}{suffix}.md", metadata, results, flags
    )

    for f in flags:
        print(f"LEAKAGE: {f}", flush=True)

    if use_wandb:
        import wandb

        for r in results:
            tag = f"{r['model']}/{r['task']}/port={r['dst_port']}"
            run_flags = [f for f in flags if tag in f]
            with wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                job_type="baseline",
                name=f"{r['model']}-{r['task']}-port{r['dst_port']}"
                + ("-dedup" if drop_duplicates else ""),
                config={
                    "model": r["model"],
                    "task": r["task"],
                    "dst_port": r["dst_port"],
                    "n_features": r["n_features"],
                    "seed": SEED,
                    "git_commit": metadata["git_commit"],
                    "data_version": metadata["data"]["version"],
                    "split": eval_split,
                    "drop_duplicates": drop_duplicates,
                },
                reinit="finish_previous",
            ) as run:
                run.log(
                    {
                        "macro_f1": r["macro_f1"],
                        "benign_fpr": r["benign_fpr"],
                    }
                )
                run.summary["leakage_flagged"] = bool(run_flags)
                if run_flags:
                    run.summary["leakage_flags"] = run_flags

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEN-109 sanity baselines")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--eval-split", default=EVAL_SPLIT, choices=["val", "test"])
    parser.add_argument("--wandb", action="store_true", help="log to W&B")
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="drop exact full-row duplicates in train and eval splits independently",
    )
    args = parser.parse_args()

    run_baselines(
        data_dir=args.data_dir,
        eval_split=args.eval_split,
        use_wandb=args.wandb,
        train_sample=args.train_sample,
        drop_duplicates=args.drop_duplicates,
    )

"""SEN-109 / SEN-FR-07: Sanity baselines (majority class + logistic regression) on data v1.

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

from sentinel_data.split import RARE_LABEL
from sentinel_features.features import LABELS as CANONICAL_LABELS
from sentinel_ml.eval_template import (  # noqa: F401
    ID_COLS,
    SEED,
    USE_WANDB,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _model_params,
    evaluate_single,
    extract_model_params,
    find_train_twins,
    get_code_commit,
    get_data_metadata,
    get_wandb_init_kwargs,
    prep_xy,
)

DATA_DIR = Path("data/processed")
EVAL_SPLIT = "test"

LEAK_LOGREG_THRESHOLD = 0.95
LEAK_ANY_THRESHOLD = 0.999


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _data_metadata(data_dir: Path) -> dict:
    card = Path(data_dir) / "data_card.json"
    if not card.exists():
        return {"version": "unknown", "rare_class_policy": "unknown"}
    d = json.loads(card.read_text())
    rare = d.get("rare_classes", {}).get("merged") or []
    return {
        "version": d.get("version", "unknown"),
        "rare_class_policy": (
            f"raw labels with <{d.get('rare_classes', {}).get('min_flows', '?')} "
            f"flows merged into '{RARE_LABEL}' by sentinel_data.split (SEN-FR-05)"
        ),
        "rare_classes_merged": sorted(rare),
    }


def canonicalize_labels(
    df: pd.DataFrame,
    canonical_labels: list[str] = CANONICAL_LABELS,
) -> pd.DataFrame:
    """Guard: verify ``df['label']`` matches the canonical class set exactly.

    The rare-class merge is owned by ``sentinel_data.split`` (SEN-FR-05); this
    function does not merge. If any raw (unmerged) label from CICIDS is
    present, the parquet was written before ``split.merge_rare_classes`` ran
    and must be regenerated. In that case we raise with instructions, rather
    than silently merging and masking stale data.
    """
    actual = set(df["label"].astype(str).unique())
    expected = set(canonical_labels)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra or missing:
        raise ValueError(
            f"Multi-class label set does not match canonical "
            f"({len(canonical_labels)} classes). "
            f"Extra: {extra or 'none'}. Missing: {missing or 'none'}. "
            f"If 'Extra' lists raw CICIDS labels (e.g. Heartbleed, "
            f"Web Attack - XSS), the parquet is stale: re-run "
            f"`python -m sentinel_data.split`."
        )
    return df


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


def _build_models(seed: int = SEED):
    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
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
    use_wandb: bool = USE_WANDB,
    out_dir: Path = Path("results"),
    train_sample: int | None = None,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    drop_duplicates: bool = False,
    canonical_labels: list[str] | None = None,
    seed: int = SEED,
) -> list[dict]:
    """Run majority + logreg baselines across binary/multiclass and with/without dst_port.

    Metric computation uses sentinel_ml.eval_template.evaluate_single so results match
    the team template format consumed by SEN-108.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    data_meta = get_data_metadata(data_dir)
    commit_hash = get_code_commit()

    train_df = pd.read_parquet(data_dir / "train.parquet")
    if train_sample is not None:
        train_df = train_df.sample(
            n=min(train_sample, len(train_df)), random_state=seed
        )
    eval_df = pd.read_parquet(data_dir / f"{eval_split}.parquet")

    # SEN-FR-05: the rare-class merge is done by sentinel_data.split. Here we
    # only *verify* train/eval already match the canonical 12 classes; stale
    # parquet with raw CICIDS labels raises instead of being silently merged.
    labels = canonical_labels if canonical_labels is not None else CANONICAL_LABELS
    canonicalize_labels(train_df, canonical_labels=labels)
    canonicalize_labels(eval_df, canonical_labels=labels)
    print(
        f"label-set guard passed: {len(labels)} canonical classes",
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

    model_factories = _build_models(seed=seed)
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
            elapsed = round(time.time() - t0, 2)
            if name == "majority":
                # DummyClassifier has no coef_/feature_importances_, so evaluate_single
                # would fall back to a very slow permutation_importance pass. Feature
                # importance for a constant predictor is meaningless; short-circuit it.
                model.feature_importances_ = np.zeros(X_ev.shape[1])
            metrics = evaluate_single(model, X_ev, y_ev, benign_name, X_train=X_tr, y_train=y_tr)
            tw = metrics.get("train_twin_analysis", {})
            metrics.update(
                {
                    "model": name,
                    "task": task,
                    "dst_port": use_port,
                    "eval_split": eval_split,
                    "n_features": int(X_ev.shape[1]),
                    "seed": seed,
                    "train_rows": int(len(X_tr)),
                    "eval_rows": int(len(X_ev)),
                    "data_version": data_meta["data_version"],
                    "code_commit": commit_hash,
                    "model_params": _model_params(model),
                    "fit_time_sec": elapsed,
                    "macro_f1_unseen": tw.get("macro_f1_unseen"),
                    "benign_fpr_unseen": tw.get("benign_fpr_unseen"),
                    "delta_macro_f1_unseen_minus_all": tw.get("delta_macro_f1_unseen_minus_all"),
                    "pct_feature_twins": tw.get("pct_feature_twins"),
                }
            )
            print(
                f"  done in {elapsed:.1f}s | "
                f"F1={metrics['macro_f1']:.4f} | FPR={metrics['benign_fpr']:.4f}",
                flush=True,
            )
            results.append(metrics)

    flags = leakage_flags(results)
    metadata = {
        "sen_ticket": "SEN-109",
        "seed": seed,
        "git_commit": _git_commit(),
        "data": _data_metadata(data_dir),
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
            run_name = f"baseline_{r['model']}_{r['task']}_{'port' if r['dst_port'] else 'noport'}_{eval_split}"
            if drop_duplicates:
                run_name += "_dedup"
            run_config = {
                **data_meta,
                "model": r["model"],
                "task": r["task"],
                "dst_port": r["dst_port"],
                "eval_split": r["eval_split"],
                "split": eval_split,
                "n_features": r["n_features"],
                "seed": r["seed"],
                "git_commit": metadata["git_commit"],
                "code_commit": r["code_commit"],
                "data_version": r["data_version"],
                "train_rows": r["train_rows"],
                "eval_rows": r["eval_rows"],
                "drop_duplicates": drop_duplicates,
                "parameters": r["parameters"],
                "model_params": r["model_params"],
            }
            init_kwargs = get_wandb_init_kwargs(job_type="baseline", name=run_name, config=run_config)
            init_kwargs["project"] = wandb_project
            if init_kwargs.get("mode") == "online":
                init_kwargs["entity"] = wandb_entity
            with wandb.init(**init_kwargs) as run:
                log_payload = {
                    "macro_f1": r["macro_f1"],
                    "weighted_f1": r["weighted_f1"],
                    "accuracy": r["accuracy"],
                    "benign_fpr": r["benign_fpr"],
                    "macro_f1_unseen": r["macro_f1_unseen"],
                    "benign_fpr_unseen": r["benign_fpr_unseen"],
                    "delta_macro_f1_unseen_minus_all": r["delta_macro_f1_unseen_minus_all"],
                    "pct_feature_twins": r["pct_feature_twins"],
                    "fit_time_sec": r["fit_time_sec"],
                }
                for c_label, c_stats in r["per_class"].items():
                    if isinstance(c_stats, dict) and "f1-score" in c_stats:
                        log_payload[f"per_class_f1/{c_label}"] = c_stats["f1-score"]
                        log_payload[f"per_class_recall/{c_label}"] = c_stats["recall"]
                        log_payload[f"per_class_precision/{c_label}"] = c_stats["precision"]
                run.log(log_payload)
                if hasattr(run, "summary") and isinstance(run.summary, dict):
                    run.summary["leakage_flagged"] = bool(run_flags)
                    if run_flags:
                        run.summary["leakage_flags"] = run_flags

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEN-109 sanity baselines")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--eval-split", default=EVAL_SPLIT, choices=["val", "test"])
    parser.add_argument("--wandb", action="store_true", default=USE_WANDB, help="log to W&B")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="disable W&B logging")
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

"""Replay saved baseline results to W&B without re-fitting models.

Reads results/baselines_<split>[_dedup].json + the matching _meta.json and
pushes one W&B run per result row. Safe to run any time after run_baselines
has produced the JSON artifacts.
"""
import argparse
import json
from pathlib import Path

import wandb


def replay(results_path: Path, meta_path: Path, entity: str, project: str) -> None:
    results = json.loads(results_path.read_text())
    meta = json.loads(meta_path.read_text())
    dedup = meta.get("drop_duplicates", False)

    for r in results:
        tag = f"{r['model']}/{r['task']}/port={r['dst_port']}"
        run_flags = [f for f in meta.get("leakage_flags", []) if tag in f]
        name = f"{r['model']}-{r['task']}-port{r['dst_port']}" + ("-dedup" if dedup else "")
        with wandb.init(
            entity=entity,
            project=project,
            job_type="baseline",
            name=name,
            config={
                "model": r["model"],
                "task": r["task"],
                "dst_port": r["dst_port"],
                "n_features": r["n_features"],
                "seed": meta["seed"],
                "git_commit": meta["git_commit"],
                "data_version": meta["data"]["version"],
                "split": meta["split_evaluated"],
                "drop_duplicates": dedup,
            },
            reinit="finish_previous",
        ) as run:
            run.log({"macro_f1": r["macro_f1"], "benign_fpr": r["benign_fpr"]})
            run.summary["leakage_flagged"] = bool(run_flags)
            if run_flags:
                run.summary["leakage_flags"] = run_flags
        print(f"logged {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--entity", default="wcs-sentinel")
    p.add_argument("--project", default="baselines-v1")
    args = p.parse_args()
    replay(args.results, args.meta, args.entity, args.project)

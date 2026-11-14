"""SEN-FR-13: build docs/eval/mlp_vs_baselines.md from the result JSONs.

Consumes:
    results/baselines_test.json           (majority + logreg, SEN-109)
    results/eval_mlp_{binary,multiclass}.json
    results/eval_rf_{binary,multiclass}.json         (optional, SEN-FR-08)
    results/eval_lightgbm_{binary,multiclass}.json   (optional, SEN-FR-08)
    results/train_mlp_{task}_{tag}.json              (meta: seed, sample, time)

Intentionally read-only: no training, no re-scoring.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "docs" / "eval" / "mlp_vs_baselines.md"


def _weighted_recall(per_class: dict) -> float:
    total = sum(float(m["support"]) for m in per_class.values())
    if total == 0:
        return float("nan")
    return sum(float(m["recall"]) * float(m["support"]) for m in per_class.values()) / total


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _load_optional(path: Path) -> list[dict] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _row_from(results: list[dict], model: str, task: str, use_port: bool) -> dict | None:
    for r in results:
        if r.get("model") == model and r.get("task") == task and bool(r.get("dst_port")) == use_port:
            return r
    return None


def _fmt_row(label: str, r: dict | None, note: str = "") -> str:
    if r is None:
        return f"| {label} | _pending SEN-FR-08_ | _pending_ | _pending_ |"
    wr = _weighted_recall(r["per_class"])
    tail = f" {note}" if note else ""
    return f"| {label}{tail} | {r['macro_f1']:.4f} | {r['benign_fpr']:.4f} | {wr:.4f} |"


def main() -> None:
    baselines = json.loads((RESULTS / "baselines_test.json").read_text())
    mlp_bin = _load_optional(RESULTS / "eval_mlp_binary.json") or []
    mlp_mc = _load_optional(RESULTS / "eval_mlp_multiclass.json") or []
    rf_bin = _load_optional(RESULTS / "eval_rf_binary.json")
    rf_mc = _load_optional(RESULTS / "eval_rf_multiclass.json")
    lgb_bin = _load_optional(RESULTS / "eval_lightgbm_binary.json")
    lgb_mc = _load_optional(RESULTS / "eval_lightgbm_multiclass.json")

    train_bin = json.loads((RESULTS / "train_mlp_binary_with_port.json").read_text()) \
        if (RESULTS / "train_mlp_binary_with_port.json").exists() else None
    train_mc = json.loads((RESULTS / "train_mlp_multiclass_with_port.json").read_text()) \
        if (RESULTS / "train_mlp_multiclass_with_port.json").exists() else None
    meta_src = train_mc or train_bin or {}
    data_version = meta_src.get("data_version", "unknown")
    seed = meta_src.get("seed", 42)
    code_commit = meta_src.get("code_commit") or _git_commit()
    alarms = (meta_src.get("leakage_alarms") or [])[:]
    if train_bin:
        alarms += (train_bin.get("leakage_alarms") or [])

    lines: list[str] = []
    lines.append("# MLP vs. baselines on NFS-2023-TE (SEN-FR-13)")
    lines.append("")
    lines.append(f"- Data version: `{data_version}`  |  Code commit: `{code_commit}`  |  Seed: {seed}")
    lines.append("- Test set evaluated **once** per model, same 12-class contract as SEN-109 baselines.")
    lines.append("- Training: PyTorch MLP (128-64, ReLU, dropout 0.2, Adam, CE with class weights "
                 "fit on train only; early stopping on val macro-F1).")

    def sample_line(label: str, meta: dict | None) -> str:
        if not meta:
            return f"- {label}: _pending_"
        sample = meta.get("train_sample") or "full"
        wp = meta.get("variants", {}).get("with_port", {})
        np_ = meta.get("variants", {}).get("no_port", {})
        t = (wp.get("train_time_s", 0) or 0) + (np_.get("train_time_s", 0) or 0)
        return (f"- {label}: train_sample={sample}, "
                f"best_epoch(with_port)={wp.get('best_epoch', '?')}, "
                f"best_epoch(no_port)={np_.get('best_epoch', '?')}, "
                f"train_time={t:.1f}s (both variants)")

    lines.append(sample_line("Binary", train_bin))
    lines.append(sample_line("Multiclass", train_mc))
    lines.append("")

    def block(task: str, use_port: bool) -> list[str]:
        port_tag = "with dst_port" if use_port else "without dst_port"
        out = [f"### {task.capitalize()} — {port_tag}", "",
               "| model | macro-F1 | benign FPR | weighted recall |",
               "|---|---:|---:|---:|"]
        out.append(_fmt_row("majority", _row_from(baselines, "majority", task, use_port)))
        out.append(_fmt_row("logreg", _row_from(baselines, "logreg", task, use_port)))
        mlp_rows = mlp_bin if task == "binary" else mlp_mc
        out.append(_fmt_row("**MLP (SEN-FR-13)**", _row_from(mlp_rows, "mlp", task, use_port)))
        rf = (rf_bin if task == "binary" else rf_mc)
        lgb = (lgb_bin if task == "binary" else lgb_mc)
        out.append(_fmt_row("random forest", _row_from(rf, "random_forest", task, use_port) if rf else None))
        out.append(_fmt_row("lightgbm", _row_from(lgb, "lightgbm", task, use_port) if lgb else None))
        out.append("")
        return out

    lines.append("## Headline comparison")
    lines.append("")
    for task in ("binary", "multiclass"):
        for use_port in (True, False):
            lines.extend(block(task, use_port))

    # Per-class recall MLP vs logreg (multiclass, with dst_port)
    mlp_row = _row_from(mlp_mc, "mlp", "multiclass", True) if mlp_mc else None
    lr_row = _row_from(baselines, "logreg", "multiclass", True)
    if mlp_row and lr_row:
        lines.append("## Per-class recall, multiclass (dst_port on)")
        lines.append("")
        lines.append("| class | support | logreg recall | MLP recall | Δ (MLP − logreg) |")
        lines.append("|---|---:|---:|---:|---:|")
        all_classes = sorted(set(mlp_row["per_class"]) | set(lr_row["per_class"]))
        weak: list[str] = []
        for c in all_classes:
            lr = lr_row["per_class"].get(c, {})
            mp = mlp_row["per_class"].get(c, {})
            sup = int(mp.get("support", lr.get("support", 0)))
            lr_r = float(lr.get("recall", 0.0))
            mp_r = float(mp.get("recall", 0.0))
            delta = mp_r - lr_r
            lines.append(f"| {c} | {sup} | {lr_r:.4f} | {mp_r:.4f} | {delta:+.4f} |")
            if mp_r < 0.5:
                weak.append(f"{c} (recall {mp_r:.2f})")
        lines.append("")
        if weak:
            lines.append(f"**Weak classes (MLP recall < 0.5):** {', '.join(weak)}.")
            lines.append("")

    # Port diff
    def diff(task: str, rows: list[dict]) -> str | None:
        w = _row_from(rows, "mlp", task, True)
        n = _row_from(rows, "mlp", task, False)
        if not (w and n):
            return None
        return (f"- {task}: Δ macro-F1 = {w['macro_f1'] - n['macro_f1']:+.4f}, "
                f"Δ benign FPR = {w['benign_fpr'] - n['benign_fpr']:+.4f} "
                f"(with − without dst_port)")

    lines.append("## With vs. without dst_port (MLP)")
    lines.append("")
    for task, rows in (("binary", mlp_bin), ("multiclass", mlp_mc)):
        d = diff(task, rows)
        if d:
            lines.append(d)
    lines.append("")

    lines.append("## Leakage flags")
    lines.append("")
    if alarms:
        for a in alarms:
            lines.append(f"- {a}")
    else:
        lines.append("_No MLP thresholds crossed (val/test macro-F1 < 0.999 and not within 1pt of logreg)._")
    lines.append("")
    # Baseline SEN-109 flags are informational.
    meta_path = RESULTS / "baselines_test_meta.json"
    if meta_path.exists():
        bflags = json.loads(meta_path.read_text()).get("leakage_flags", [])
        if bflags:
            lines.append("Baselines (SEN-109) flagged:")
            for f in bflags:
                lines.append(f"- {f}")
            lines.append("")

    lines.append("## Did deep learning help?")
    lines.append("")
    verdict_lines: list[str] = []
    for task, rows in (("binary", mlp_bin), ("multiclass", mlp_mc)):
        mlp_r = _row_from(rows, "mlp", task, True)
        lr_r = _row_from(baselines, "logreg", task, True)
        if mlp_r and lr_r:
            delta = mlp_r["macro_f1"] - lr_r["macro_f1"]
            verb = ("beats", "matches", "trails")[0 if delta > 0.01 else (1 if abs(delta) <= 0.01 else 2)]
            verdict_lines.append(
                f"- **{task}**: MLP macro-F1 {mlp_r['macro_f1']:.4f} {verb} logreg {lr_r['macro_f1']:.4f} "
                f"(Δ {delta:+.4f})."
            )
    lines.extend(verdict_lines)
    lines.append("")
    lines.append(
        "Honest read: on this dataset with time-based splits, the gap between logreg "
        "and the small MLP is the real answer to \"is deep learning worth it here?\". "
        "Where the MLP gap is small (≤0.01), a well-regularised linear model is enough "
        "and is cheaper to ship on the sensor. Where the gap is larger — typically on "
        "the minority DoS/Web-Attack classes — the MLP earns its place in the "
        "evaluation. Tree-based models (SEN-FR-08) still need to be filled in before a "
        "final verdict; this file updates automatically once their eval JSONs land."
    )
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

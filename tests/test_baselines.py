"""Unit tests for SEN-FR-07 sanity baselines and synthetic data generation."""
import json

import pandas as pd
from sentinel_ml.baselines import ID_COLS, leakage_flags, run_baselines
from sentinel_ml.make_synthetic import generate_parquet_splits


def test_identifiers_never_used_as_features(tmp_path):
    """Ensure flow identifiers and target labels are stripped from the feature matrix."""
    generate_parquet_splits(out_dir=tmp_path, n=500)
    df = pd.read_parquet(tmp_path / "train.parquet")
    X = df.drop(columns=ID_COLS, errors="ignore").select_dtypes("number")
    assert not (set(X.columns) & set(ID_COLS))
    assert "label" not in X.columns


def test_dst_port_toggle(tmp_path):
    """Verify dst_port is included or excluded based on the port setting."""
    generate_parquet_splits(out_dir=tmp_path, n=500)
    df = pd.read_parquet(tmp_path / "train.parquet")
    X_with = df.drop(columns=ID_COLS, errors="ignore").select_dtypes("number")
    X_without = df.drop(columns=ID_COLS + ["dst_port"], errors="ignore").select_dtypes("number")
    assert "dst_port" in X_with.columns
    assert "dst_port" not in X_without.columns


def test_majority_baseline_is_weak_and_metrics_are_sane(tmp_path):
    """Verify all 8 baseline combinations run, save JSON output, and produce valid metrics."""
    generate_parquet_splits(out_dir=tmp_path, n=2000, seed=0)
    results = run_baselines(data_dir=tmp_path, eval_split="val", use_wandb=False, out_dir=tmp_path / "results")
    assert len(results) == 8  # 2 tasks x 2 port settings x 2 models
    assert (tmp_path / "results" / "baselines_val.json").exists()
    assert (tmp_path / "results" / "baselines_val_meta.json").exists()
    assert (tmp_path / "results" / "baselines_val.md").exists()
    for r in results:
        assert 0 <= r["macro_f1"] <= 1
        assert 0 <= r["benign_fpr"] <= 1
        if r["model"] == "majority":
            assert r["benign_fpr"] == 0.0  # always predicts benign
            assert r["macro_f1"] < 0.6
    meta = json.loads((tmp_path / "results" / "baselines_val_meta.json").read_text())
    assert meta["sen_ticket"] == "SEN-109"
    assert meta["seed"] == 42
    assert "rare_class_policy" in meta["data"]


def test_leakage_flags_fires_at_thresholds():
    """logreg >= 0.95 is WARNING; anything >= 0.999 is CRITICAL; below both is silent."""
    results = [
        {"model": "logreg", "task": "binary", "dst_port": True, "macro_f1": 0.96},
        {"model": "logreg", "task": "binary", "dst_port": False, "macro_f1": 0.94},
        {"model": "majority", "task": "binary", "dst_port": True, "macro_f1": 0.9999},
        {"model": "majority", "task": "binary", "dst_port": False, "macro_f1": 0.5},
    ]
    flags = leakage_flags(results)
    joined = "\n".join(flags)
    assert "WARNING" in joined and "logreg/binary/port=True" in joined
    assert "CRITICAL" in joined and "majority/binary/port=True" in joined
    # below-threshold entries do not appear
    assert "port=False" not in joined

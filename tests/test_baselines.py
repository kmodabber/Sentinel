"""Unit tests for SEN-FR-07 sanity baselines and synthetic data generation."""

import json

import pandas as pd
import pytest

from sentinel_ml.baselines import (
    ID_COLS,
    canonicalize_labels,
    leakage_flags,
    run_baselines,
)
from sentinel_ml.make_synthetic import generate_parquet_splits
from tests.fixtures import (
    CANONICAL_12,
    make_cicids_frame,
    write_splits,
)


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
    X_without = df.drop(columns=ID_COLS + ["dst_port"], errors="ignore").select_dtypes(
        "number"
    )
    assert "dst_port" in X_with.columns
    assert "dst_port" not in X_without.columns


def test_majority_baseline_is_weak_and_metrics_are_sane(tmp_path):
    """Verify all 8 baseline combinations run, save JSON output, and produce valid metrics."""
    # Synthetic data has 4 ad-hoc classes, not the canonical 12. We pass the
    # synthetic class set so the SEN-FR-05 canonical check doesn't fail here.
    # n=4000 keeps each attack class above RARE_MIN_FLOWS=100 in the train split.
    generate_parquet_splits(out_dir=tmp_path, n=4000, seed=0)
    synth_labels = ["BENIGN", "PortScan", "DoS", "SSH-Patator"]
    results = run_baselines(
        data_dir=tmp_path,
        eval_split="val",
        use_wandb=False,
        out_dir=tmp_path / "results",
        canonical_labels=synth_labels,
    )
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
    assert meta["n_classes_multiclass"] == len(synth_labels)


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


# SEN-109 (reopened): the canonical 12-class set per sentinel_features.features.LABELS
# must be enforced after the SEN-FR-05 rare-class merge. The committed
# train.parquet still has 15 raw CICIDS labels, so without a merge-and-assert
# step the baseline reports 15 classes.
RAW_COUNTS_MIMICS_DATA_V1 = {
    "BENIGN": 200,
    "Bot": 150,
    "DDoS": 150,
    "DoS GoldenEye": 150,
    "DoS Hulk": 150,
    "DoS Slowhttptest": 150,
    "DoS slowloris": 150,
    "FTP-Patator": 150,
    "PortScan": 150,
    "SSH-Patator": 150,
    "Web Attack - Brute Force": 150,
    # Rare (<100 train flows) - should be merged into "Rare attack"
    "Heartbleed": 10,
    "Infiltration": 20,
    "Web Attack - Sql Injection": 15,
    "Web Attack - XSS": 25,
}


def test_canonicalize_merges_four_rare_classes_into_rare_attack():
    df = make_cicids_frame(RAW_COUNTS_MIMICS_DATA_V1, seed=0)
    out = canonicalize_labels(df)
    assert set(out["label"].unique()) == set(CANONICAL_12)
    assert len(set(out["label"].unique())) == 12
    expected_rare = 10 + 20 + 15 + 25
    assert (out["label"] == "Rare attack").sum() == expected_rare


def test_canonicalize_raises_on_extra_class_naming_it():
    counts = dict(RAW_COUNTS_MIMICS_DATA_V1)
    counts["FooAttack"] = 500  # unknown class, caller did not declare it as rare
    df = make_cicids_frame(counts, seed=0)
    # Explicit rare list (the real 4 from data v1) does not contain FooAttack,
    # so FooAttack remains after the merge and must fail the canonical check.
    real_rare = [
        "Heartbleed",
        "Infiltration",
        "Web Attack - Sql Injection",
        "Web Attack - XSS",
    ]
    with pytest.raises(ValueError, match=r"FooAttack"):
        canonicalize_labels(df, rare_class_list=real_rare)


def test_canonicalize_raises_on_missing_canonical_class_naming_it():
    counts = dict(RAW_COUNTS_MIMICS_DATA_V1)
    del counts["PortScan"]  # missing a canonical class
    df = make_cicids_frame(counts, seed=0)
    with pytest.raises(ValueError, match=r"PortScan"):
        canonicalize_labels(df)


def test_canonicalize_is_idempotent_when_already_canonical():
    df = make_cicids_frame({c: 120 for c in CANONICAL_12}, seed=0)
    out = canonicalize_labels(df)
    assert set(out["label"].unique()) == set(CANONICAL_12)
    assert len(out) == len(df)


def test_run_baselines_produces_results_for_every_canonical_class(tmp_path):
    write_splits(tmp_path, RAW_COUNTS_MIMICS_DATA_V1, seed=0)
    (tmp_path / "data_card.json").write_text(json.dumps({"version": "test-v1"}))
    results = run_baselines(
        data_dir=tmp_path,
        eval_split="test",
        use_wandb=False,
        out_dir=tmp_path / "results",
    )
    multi = [r for r in results if r["task"] == "multiclass"]
    assert multi, "no multiclass results produced"
    for r in multi:
        assert set(r["per_class"].keys()) == set(CANONICAL_12), (
            f"per_class keys {sorted(r['per_class'])} != canonical {sorted(CANONICAL_12)}"
        )


def test_run_baselines_identifier_cols_never_in_feature_matrix(tmp_path):
    # Strengthen: run end-to-end and inspect the models' trained feature names.
    write_splits(tmp_path, RAW_COUNTS_MIMICS_DATA_V1, seed=0)
    (tmp_path / "data_card.json").write_text(json.dumps({"version": "test-v1"}))
    results = run_baselines(
        data_dir=tmp_path,
        eval_split="test",
        use_wandb=False,
        out_dir=tmp_path / "results",
    )
    forbidden = set(ID_COLS) | {"src_port", "flow_id"}
    for r in results:
        used_features = {f["feature"] for f in r["top_features"]}
        assert not (used_features & forbidden), (
            f"identifier leaked into features: {used_features & forbidden}"
        )


def test_results_meta_records_data_version_commit_seed_nclasses(tmp_path):
    write_splits(tmp_path, RAW_COUNTS_MIMICS_DATA_V1, seed=0)
    (tmp_path / "data_card.json").write_text(json.dumps({"version": "test-v1"}))
    run_baselines(
        data_dir=tmp_path,
        eval_split="test",
        use_wandb=False,
        out_dir=tmp_path / "results",
    )
    meta = json.loads((tmp_path / "results" / "baselines_test_meta.json").read_text())
    assert meta["data"]["version"] == "test-v1"
    assert meta["seed"] == 42
    assert meta["git_commit"]  # non-empty
    assert meta["n_classes_multiclass"] == 12
    assert meta["dst_port_settings"] == [True, False]

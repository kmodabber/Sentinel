"""Unit tests for the universal evaluation template (eval_template.py)."""
import json
import sys
import types

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel_ml.eval_template import (
    SEED,
    evaluate_single,
    get_top_features,
    prep_xy,
    run_eval_template,
)
from sentinel_ml.make_synthetic import generate_parquet_splits


def test_blueprint_auto_fits_with_and_without_port(tmp_path):
    """Verify passing an unfitted estimator blueprint trains and evaluates both port variants."""
    generate_parquet_splits(out_dir=tmp_path, n=1000, seed=0)
    rf = RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42)
    results = run_eval_template(
        rf,
        data_dir=tmp_path,
        task="multiclass",
        model_name="random_forest",
        out_dir=tmp_path / "results",
    )
    assert len(results) == 2
    assert results[0]["dst_port"] is True
    assert results[1]["dst_port"] is False
    assert results[0]["n_features"] == results[1]["n_features"] + 1
    assert (tmp_path / "results" / "eval_random_forest_multiclass.json").exists()
    for r in results:
        assert 0 <= r["macro_f1"] <= 1
        assert 0 <= r["benign_fpr"] <= 1
        assert len(r["top_features"]) > 0
        assert r["seed"] == SEED == 42
        assert r["code_commit"]
        assert "data_version" in r


def test_pretrained_pair_inference_only(tmp_path):
    """Verify passing a dictionary of pre-trained models runs test inference without refitting."""
    generate_parquet_splits(out_dir=tmp_path, n=1000, seed=0)
    train_df = pd.read_parquet(tmp_path / "train.parquet")

    X_tr_port, y_tr_port, _ = prep_xy(train_df, use_port=True, task="binary")
    X_tr_noport, y_tr_noport, _ = prep_xy(train_df, use_port=False, task="binary")

    m_with = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(X_tr_port, y_tr_port)
    m_no = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(X_tr_noport, y_tr_noport)

    joblib.dump(m_with, tmp_path / "m_with.joblib")
    joblib.dump(m_no, tmp_path / "m_no.joblib")

    trained_pair = {
        "with_port": joblib.load(tmp_path / "m_with.joblib"),
        "no_port": joblib.load(tmp_path / "m_no.joblib"),
    }
    results = run_eval_template(
        trained_pair,
        data_dir=tmp_path,
        task="binary",
        model_name="logreg_pair",
        out_dir=tmp_path / "results",
    )
    assert len(results) == 2
    assert (tmp_path / "results" / "eval_logreg_pair_binary.json").exists()


def test_three_tier_feature_importance_fallback(tmp_path):
    """Verify get_top_features works for models without feature_importances_ or coef_ via permutation."""
    generate_parquet_splits(out_dir=tmp_path, n=500, seed=0)
    df = pd.read_parquet(tmp_path / "train.parquet")
    X, y, benign_name = prep_xy(df, use_port=True, task="binary")

    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    top_feats = get_top_features(dummy, X, y, top_k=3)
    assert len(top_feats) == 3
    assert all("feature" in f and "importance" in f for f in top_feats)

    metrics = evaluate_single(dummy, X, y, benign_name, top_k=3)
    assert metrics["benign_fpr"] == 0.0


def test_run_eval_template_records_and_logs_commit_data_version_seed(tmp_path, monkeypatch):
    """Verify run_eval_template logs code_commit, data_version, seed, and model.get_params() to W&B project 'Sentinel'."""
    generate_parquet_splits(out_dir=tmp_path, n=400, seed=0)
    (tmp_path / "data_card.json").write_text(json.dumps({"version": "nfs2023te-v1"}))

    init_calls = []

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, payload):
            pass

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: (init_calls.append(kwargs), DummyRun())[1]
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    dummy = DummyClassifier(strategy="most_frequent")
    results = run_eval_template(
        dummy,
        data_dir=tmp_path,
        task="binary",
        model_name="dummy",
        out_dir=tmp_path / "results",
        use_wandb=True,
    )
    assert len(init_calls) == 2
    for call, r in zip(init_calls, results):
        assert call["project"] == "Sentinel"
        cfg = call["config"]
        assert cfg["code_commit"] == r["code_commit"]
        assert cfg["data_version"] == "nfs2023te-v1"
        assert cfg["seed"] == 42
        assert cfg["strategy"] == "most_frequent"
        assert cfg["model_params"]["strategy"] == "most_frequent"


def test_run_eval_template_extracts_random_state_and_model_params(tmp_path, monkeypatch):
    """Verify run_eval_template pulls random_state from estimators/pipelines and logs model hyperparameters."""
    generate_parquet_splits(out_dir=tmp_path, n=400, seed=0)

    init_calls = []

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, payload):
            pass

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: (init_calls.append(kwargs), DummyRun())[1]
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=7)
    results = run_eval_template(
        rf,
        data_dir=tmp_path,
        task="binary",
        model_name="rf",
        out_dir=tmp_path / "results",
        use_wandb=True,
    )
    for call, r in zip(init_calls, results):
        assert r["seed"] == 7
        assert r["model_params"]["n_estimators"] == 15
        assert r["model_params"]["max_depth"] == 4
        assert r["model_params"]["random_state"] == 7
        assert call["config"]["seed"] == 7
        assert call["config"]["n_estimators"] == 15
        assert call["config"]["max_depth"] == 4

    # Verify Pipeline random_state resolution as well
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=13))
    pipe_results = run_eval_template(
        pipe,
        data_dir=tmp_path,
        task="binary",
        model_name="logreg_pipe",
        out_dir=tmp_path / "results",
    )
    for r in pipe_results:
        assert r["seed"] == 13
        assert r["model_params"]["max_iter"] == 500
        assert r["model_params"]["random_state"] == 13


def test_find_twin_mask_identifies_exact_feature_and_label_twins(tmp_path):
    """Verify scripts/check_train_test_twins.py detects exact (features, label) twins across splits."""
    from scripts.check_train_test_twins import find_twin_mask

    generate_parquet_splits(out_dir=tmp_path, n=300, seed=0)
    train_df = pd.read_parquet(tmp_path / "train.parquet")
    test_df = pd.read_parquet(tmp_path / "test.parquet").copy()

    # Copy 5 train rows into test_df with different IDs/timestamps so only features+label match
    injected = train_df.iloc[:5].copy()
    if "bidirectional_first_seen_ms" in injected.columns:
        injected["bidirectional_first_seen_ms"] += 999_999
    test_with_twins = pd.concat([injected, test_df], ignore_index=True)

    mask, feat_cols = find_twin_mask(train_df, test_with_twins, use_port=True)
    assert "dst_port" in feat_cols
    assert mask[:5].all()


def test_train_mlp_propagates_custom_seed_to_eval_json(tmp_path):
    """Verify train_mlp --seed 7 writes seed=7 to eval_mlp_binary.json."""
    from sentinel_features.features import LABELS as CANONICAL_LABELS
    from sentinel_ml.train_mlp import main as train_main

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    generate_parquet_splits(out_dir=data_dir, n=300, seed=0)
    # train_mlp.main runs strict canonicalize_labels, so each split must contain
    # exactly the canonical label set. Overwrite labels on the first N rows.
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        path = data_dir / name
        df = pd.read_parquet(path)
        labels = list(df["label"].to_numpy())
        for i, lab in enumerate(CANONICAL_LABELS):
            labels[i] = lab
        df["label"] = labels
        df.to_parquet(path)

    train_main(
        task="binary",
        data_dir=data_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        epochs=1,
        patience=1,
        batch_size=64,
        hidden=(8, 4),
        seed=7,
        use_wandb=False,
    )
    rows = json.loads((results_dir / "eval_mlp_binary.json").read_text())
    assert len(rows) == 2
    for r in rows:
        assert r["seed"] == 7
        assert r["model_params"]["hidden"] == [8, 4]
        assert r["model_params"]["seed"] == 7

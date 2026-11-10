"""Unit tests for the universal evaluation template (eval_template.py)."""
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel_ml.eval_template import (
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

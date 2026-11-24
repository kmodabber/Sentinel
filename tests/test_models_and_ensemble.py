"""Unit tests for Random Forest, LightGBM, Fixed-Weights Ensemble, and Meta-Learner Ensemble."""
import numpy as np
import pandas as pd

from sentinel_ml.ensemble import (
    FixedWeightEnsemble,
    MetaLearnerEnsemble,
    default_lgbm,
    default_rf,
    sweep_fixed_weights,
)
from sentinel_ml.eval_template import evaluate_single, prep_xy, run_eval_template
from sentinel_ml.make_synthetic import generate_parquet_splits
from sentinel_ml.train_and_compare import run_training_and_comparison


def test_rf_and_lgbm_with_eval_template(tmp_path):
    """Verify both Random Forest and LightGBM train and evaluate via run_eval_template."""
    generate_parquet_splits(out_dir=tmp_path, n=800, seed=0)

    rf = default_rf(task="binary", n_estimators=15)
    lgbm = default_lgbm(task="binary", n_estimators=15)

    res_rf = run_eval_template(rf, data_dir=tmp_path, task="binary", model_name="random_forest", out_dir=tmp_path / "results")
    res_lgbm = run_eval_template(lgbm, data_dir=tmp_path, task="binary", model_name="lightgbm", out_dir=tmp_path / "results")

    assert len(res_rf) == 2
    assert len(res_lgbm) == 2
    assert (tmp_path / "results" / "eval_random_forest_binary.json").exists()
    assert (tmp_path / "results" / "eval_lightgbm_binary.json").exists()
    for r in res_rf + res_lgbm:
        assert 0.0 <= r["macro_f1"] <= 1.0
        assert 0.0 <= r["benign_fpr"] <= 1.0
        assert len(r["top_features"]) > 0


def test_fixed_weight_and_metalearner_ensembles(tmp_path):
    """Verify FixedWeightEnsemble and MetaLearnerEnsemble work both from scratch and from pre-fitted models."""
    generate_parquet_splits(out_dir=tmp_path, n=1000, seed=42)
    train_df = pd.read_parquet(tmp_path / "train.parquet")
    val_df = pd.read_parquet(tmp_path / "val.parquet")
    test_df = pd.read_parquet(tmp_path / "test.parquet")

    for task in ("binary", "multiclass"):
        X_tr, y_tr, benign_name = prep_xy(train_df, use_port=True, task=task)
        X_va, y_va, _ = prep_xy(val_df, use_port=True, task=task)
        X_te, y_te, _ = prep_xy(test_df, use_port=True, task=task)

        rf = default_rf(task=task, n_estimators=15).fit(X_tr, y_tr)
        lgbm = default_lgbm(task=task, n_estimators=15).fit(X_tr, y_tr)

        fixed = FixedWeightEnsemble.from_prefitted(rf, lgbm, weights=(0.5, 0.5), task=task)
        meta = MetaLearnerEnsemble.from_prefitted(rf, lgbm, X_va, y_va, task=task)

        probs_fixed = fixed.predict_proba(X_te)
        probs_meta = meta.predict_proba(X_te)
        assert probs_fixed.shape == probs_meta.shape
        assert np.allclose(probs_fixed.sum(axis=1), 1.0, atol=1e-4)
        assert np.allclose(probs_meta.sum(axis=1), 1.0, atol=1e-4)

        m_fixed = evaluate_single(fixed, X_te, y_te, benign_name, top_k=4)
        m_meta = evaluate_single(meta, X_te, y_te, benign_name, top_k=4)
        assert 0.0 <= m_fixed["macro_f1"] <= 1.0
        assert 0.0 <= m_meta["macro_f1"] <= 1.0
        assert len(m_fixed["top_features"]) == 4
        assert len(m_meta["top_features"]) == 4

        contrib = meta.get_model_contributions()
        assert 0.0 <= contrib["rf_share"] <= 1.0
        assert 0.0 <= contrib["lgbm_share"] <= 1.0
        assert abs(contrib["rf_share"] + contrib["lgbm_share"] - 1.0) < 1e-3

        sweep = sweep_fixed_weights(
            rf.predict_proba(X_va),
            lgbm.predict_proba(X_va),
            np.asarray(y_va),
            np.asarray(rf.classes_),
            benign_name,
            steps=5,
        )
        assert len(sweep["grid"]) == 5
        assert abs(sum(sweep["best_weights"]) - 1.0) < 1e-4


def test_full_training_and_comparison_pipeline(tmp_path, monkeypatch):
    """Verify run_training_and_comparison writes all 8 eval JSONs and 2 comparison JSONs and logs to W&B offline."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    generate_parquet_splits(out_dir=tmp_path, n=900, seed=7)

    out_dir = tmp_path / "results"
    summary = run_training_and_comparison(
        data_dir=tmp_path,
        out_dir=out_dir,
        use_wandb=True,
        n_estimators_rf=10,
        n_estimators_lgbm=10,
    )

    expected_files = [
        "eval_random_forest_binary.json",
        "eval_random_forest_multiclass.json",
        "eval_rf_binary.json",
        "eval_rf_multiclass.json",
        "eval_lightgbm_binary.json",
        "eval_lightgbm_multiclass.json",
        "eval_ensemble_fixed_binary.json",
        "eval_ensemble_fixed_multiclass.json",
        "eval_ensemble_metalearner_binary.json",
        "eval_ensemble_metalearner_multiclass.json",
        "rf_vs_lightgbm_comparison.json",
        "ensemble_comparison.json",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing output file: {fname}"

    import json
    for fname in expected_files[:10]:
        recs = json.loads((out_dir / fname).read_text())
        assert len(recs) == 2
        for r in recs:
            assert r["seed"] == 42
            assert isinstance(r["model_params"], dict)
            assert isinstance(r["parameters"], dict)
            assert r["val_tuning"]["split"] == "val"
            assert r["val_tuning"]["test_touched_during_tuning"] is False

    assert len(summary["rf_vs_lightgbm"]) == 4
    assert len(summary["ensemble_comparison"]) == 4
    assert len(summary["summary_rows"]) == 20


def test_eval_report_v1_contains_all_sen108_sections():
    """Verify docs/eval/eval_report_v1.md builds cleanly from results/*.json.

    `data/processed/data_card.json` is gitignored (data/*), so build_report() uses its
    embedded nfs2023te-v1 fallback when the card is absent in CI.
    """
    from scripts.build_eval_report_v1 import OUT_PATH, build_report

    text = build_report()
    assert OUT_PATH.exists()
    assert OUT_PATH.read_text(encoding="utf-8") == text
    for required in (
        "## 1. Headline Test-Set Comparison on Data v1",
        "## 2. Per-Class Precision, Recall, F1, Support, Benign FPR, and Confusion Matrices",
        "## 3. `Rare attack` Class Merge Quoted from `data/processed/data_card.json` (`SEN-FR-05`)",
        "## 4. Leakage Check Result (`SEN-106`) & Weights & Biases Run Links (`SEN-107`)",
        "## 5. Provisional Stage 2 Quality Targets Confirmed or Changed with Kam",
        "attack classes with fewer than 100 flows after cleaning are merged into 'Rare attack'",
        "https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/k3gmz3iz",
    ):
        assert required in text

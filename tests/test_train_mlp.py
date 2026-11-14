"""Unit tests for SEN-FR-13 MLP trainer."""
# isort: off
# torch must import before numpy on Windows (OpenMP clash: c10.dll WinError 1114).
import torch  # noqa: F401,E402
# isort: on

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

from sentinel_features.features import LABELS as CANONICAL_LABELS
from sentinel_ml.train_mlp import (
    BINARY_LABELS,
    build_preprocessor,
    class_weights_from_train,
    task_label_set,
)
from sentinel_ml.train_mlp import main as train_main
from tests.fixtures import CANONICAL_12, RAW_CICIDS_LABELS, write_splits


def test_preprocessor_fits_on_train_and_handles_negatives():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({
        "a": rng.normal(0, 5, size=800),
        "b": rng.normal(-3, 2, size=800),
        "c": rng.exponential(1.0, size=800) - 0.5,
    }).to_numpy(dtype=np.float32)
    pre = build_preprocessor(len(X_train)).fit(X_train)
    Xt = pre.transform(X_train)
    assert np.all(np.isfinite(Xt))
    assert Xt.shape == X_train.shape
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=0.1)

    X_other = pd.DataFrame({
        "a": rng.normal(100, 1, size=400),
        "b": rng.normal(-100, 1, size=400),
        "c": rng.normal(50, 1, size=400),
    }).to_numpy(dtype=np.float32)
    quantile_before = pre.named_steps["quantile"].quantiles_.copy()
    scaler_mean_before = pre.named_steps["scaler"].mean_.copy()
    Xt_other = pre.transform(X_other)
    assert np.all(np.isfinite(Xt_other))
    assert np.array_equal(pre.named_steps["quantile"].quantiles_, quantile_before)
    assert np.array_equal(pre.named_steps["scaler"].mean_, scaler_mean_before)


def test_class_weights_match_sklearn():
    y = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    got = class_weights_from_train(y, n_classes=3)
    expected = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    assert np.allclose(got, expected)


def test_mlp_classes_match_baselines_canonical():
    """The MLP's multiclass label set must be the canonical 12 classes from
    sentinel_features.features.LABELS, not whatever happens to be in train.
    Keeps comparison with baselines meaningful (SEN-FR-07 / SEN-109)."""
    assert task_label_set("multiclass") == list(CANONICAL_LABELS)
    assert task_label_set("multiclass") == CANONICAL_12
    assert task_label_set("binary") == BINARY_LABELS
    assert set(task_label_set("binary")) == {"benign", "attack"}


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_end_to_end_tiny(tmp_path: Path, task: str):
    """Small CICIDS-shaped fixture covering all 15 raw labels (so rare classes
    exist and get merged). Checks end-to-end run produces canonical results."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    label_counts = {label: 30 for label in RAW_CICIDS_LABELS}
    write_splits(data_dir, label_counts, seed=0)

    train_main(
        task=task,
        data_dir=data_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        epochs=2,
        patience=2,
        batch_size=64,
        hidden=(16, 8),
        train_sample=None,
        use_wandb=False,
    )

    eval_file = results_dir / f"eval_mlp_{task}.json"
    assert eval_file.exists(), f"missing {eval_file}"
    rows = json.loads(eval_file.read_text())
    assert len(rows) == 2  # with and without dst_port
    expected_classes = set(task_label_set(task))
    for r in rows:
        assert 0.0 <= r["macro_f1"] <= 1.0
        assert 0.0 <= r["benign_fpr"] <= 1.0
        assert set(r["per_class"].keys()) == expected_classes

    for tag in ("with_port", "no_port"):
        assert (models_dir / f"preprocess_mlp_{task}_{tag}.joblib").exists()
        assert (models_dir / f"mlp_{task}_{tag}.pt").exists()
        assert (results_dir / f"train_mlp_{task}_{tag}.json").exists()

"""Unit tests for SEN-FR-13 MLP trainer. Uses only the synthetic fixture."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

from sentinel_ml.make_synthetic import generate_parquet_splits
from sentinel_ml.train_mlp import (
    build_preprocessor,
    class_weights_from_train,
)
from sentinel_ml.train_mlp import (
    main as train_main,
)


def test_preprocessor_fits_on_train_and_handles_negatives():
    """QuantileTransformer+StandardScaler is fitted on TRAIN only, produces finite
    outputs even with negative inputs (NFStream features can be negative), and
    its stats are frozen at transform time."""
    rng = np.random.default_rng(0)
    # Deliberately include negative values and heavy tails in the "train" set.
    X_train = pd.DataFrame({
        "a": rng.normal(0, 5, size=800),
        "b": rng.normal(-3, 2, size=800),
        "c": rng.exponential(1.0, size=800) - 0.5,
    }).to_numpy(dtype=np.float32)
    pre = build_preprocessor(len(X_train)).fit(X_train)
    Xt = pre.transform(X_train)
    assert np.all(np.isfinite(Xt))
    assert Xt.shape == X_train.shape
    # StandardScaler fitted on the quantile output: train columns ~0 mean, ~1 std.
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=0.1)

    # A very different second dataset must NOT retrain the transformer.
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


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_end_to_end_tiny(tmp_path: Path, task: str):
    """Train on a 500-row synthetic fixture for a couple of epochs, both port
    variants; check the eval-template output is produced with sane metrics."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    generate_parquet_splits(out_dir=data_dir, n=500, seed=0)

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
    import json
    rows = json.loads(eval_file.read_text())
    assert len(rows) == 2  # with and without dst_port
    for r in rows:
        assert 0.0 <= r["macro_f1"] <= 1.0
        assert 0.0 <= r["benign_fpr"] <= 1.0
        assert len(r["per_class"]) > 0

    # Per-variant training artefacts and metadata are present.
    for tag in ("with_port", "no_port"):
        assert (models_dir / f"preprocess_mlp_{task}_{tag}.joblib").exists()
        assert (models_dir / f"mlp_{task}_{tag}.pt").exists()
        assert (results_dir / f"train_mlp_{task}_{tag}.json").exists()

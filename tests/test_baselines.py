import pandas as pd
from sentinel_ml.baselines import ID_COLS, run_baselines
from sentinel_ml.make_synthetic import generate_parquet_splits


def test_identifiers_never_used_as_features(tmp_path):
    generate_parquet_splits(out_dir=tmp_path, n=500)
    df = pd.read_parquet(tmp_path / "train.parquet")
    X = df.drop(columns=ID_COLS, errors="ignore").select_dtypes("number")
    assert not (set(X.columns) & set(ID_COLS))
    assert "label" not in X.columns


def test_dst_port_toggle(tmp_path):
    generate_parquet_splits(out_dir=tmp_path, n=500)
    df = pd.read_parquet(tmp_path / "train.parquet")
    X_with = df.drop(columns=ID_COLS, errors="ignore").select_dtypes("number")
    X_without = df.drop(columns=ID_COLS + ["dst_port"], errors="ignore").select_dtypes("number")
    assert "dst_port" in X_with.columns
    assert "dst_port" not in X_without.columns


def test_majority_baseline_is_weak_and_metrics_are_sane(tmp_path):
    generate_parquet_splits(out_dir=tmp_path, n=2000, seed=0)
    results = run_baselines(data_dir=tmp_path, eval_split="val", use_wandb=False, out_dir=tmp_path / "results")
    assert len(results) == 8  # 2 tasks x 2 port settings x 2 models
    assert (tmp_path / "results" / "baselines_val.json").exists()
    for r in results:
        assert 0 <= r["macro_f1"] <= 1
        assert 0 <= r["benign_fpr"] <= 1
        if r["model"] == "majority":
            assert r["benign_fpr"] == 0.0  # always predicts benign
            assert r["macro_f1"] < 0.6

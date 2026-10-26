from sentinel_ml.baselines import ID_COLS, feature_matrix, run, synthetic_splits


def test_identifiers_never_used_as_features():
    df = synthetic_splits(n=500)["train"]
    X = feature_matrix(df, "label", use_dst_port=True)
    assert not (set(X.columns) & ID_COLS)
    assert "label" not in X.columns


def test_dst_port_toggle():
    df = synthetic_splits(n=500)["train"]
    assert "dst_port" in feature_matrix(df, "label", True).columns
    assert "dst_port" not in feature_matrix(df, "label", False).columns


def test_majority_baseline_is_weak_and_metrics_are_sane():
    results = run(synthetic_splits(n=2000), "label", "BENIGN", "val", seed=0)
    assert len(results) == 8  # 2 tasks x 2 port settings x 2 models
    for r in results:
        assert 0 <= r["macro_f1"] <= 1
        assert r["benign_fpr"] is None or 0 <= r["benign_fpr"] <= 1
        if r["model"] == "majority":
            assert r["benign_fpr"] == 0.0  # always predicts benign
            assert r["macro_f1"] < 0.6

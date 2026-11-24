"""SEN-69 tests: TreeSHAP explanations match the agreed top_reasons shape."""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401
from lightgbm import LGBMClassifier  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from sentinel_features.features import FEATURES
from sentinel_ml.dev_model import StringClassLGBM
from sentinel_ml.explain import NO_EXPLANATION_NOTE, Explainer
from sentinel_ml.feature_labels import label_for


@pytest.fixture(scope="module")
def tiny_package():
    """A tiny LightGBM + identity preprocess wrapped as a C1-ish package dict.

    Keeps to the real 55-feature-no-port shape so column indexing in the
    explainer is exercised the same way as in production.
    """
    rng = np.random.default_rng(0)
    feature_cols = [f for f in FEATURES if f != "dst_port"]  # 55
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, len(feature_cols))), columns=feature_cols)
    # Give three classes so multi-class SHAP path runs.
    classes = ["BENIGN", "PortScan", "DDoS"]
    y = np.resize(np.array([0, 1, 2], dtype=np.int32), n)
    base = LGBMClassifier(n_estimators=20, num_leaves=7, learning_rate=0.1,
                          verbose=-1, random_state=0)
    wrapper = StringClassLGBM(base)
    # Fit on the DataFrame so booster.feature_name() returns real names.
    wrapper.fit(X, np.asarray([classes[i] for i in y]))

    class IdentityPreprocess:
        def transform(self, df):
            return df

    package = {
        "version": "test-tiny-v0",
        "features": FEATURES,
        "model": wrapper,
        "preprocess": IdentityPreprocess(),
    }
    return package, X, classes


def test_explanation_has_exactly_5_reasons(tiny_package):
    package, X, classes = tiny_package
    preds = ["PortScan"] * 3
    reasons = Explainer(package).explain(X.head(3), preds)
    assert len(reasons) == 3
    for r in reasons:
        assert r["method"] == "tree_shap"
        assert r["class"] == "PortScan"
        assert r["model_version"] == "test-tiny-v0"
        assert len(r["reasons"]) == 5


def test_reasons_sorted_by_abs_shap(tiny_package):
    package, X, _ = tiny_package
    reasons = Explainer(package).explain(X.head(1), ["DDoS"])
    shap_mags = [abs(r["shap"]) for r in reasons[0]["reasons"]]
    assert shap_mags == sorted(shap_mags, reverse=True)


def test_feature_names_are_a_subset_of_model_inputs(tiny_package):
    package, X, _ = tiny_package
    reasons = Explainer(package).explain(X.head(2), ["BENIGN", "PortScan"])
    model_inputs = set([f for f in FEATURES if f != "dst_port"])
    for r in reasons:
        for reason in r["reasons"]:
            assert reason["feature"] in model_inputs
            assert reason["label"] == label_for(reason["feature"])
            assert reason["direction"] in ("up", "down")
            # value is numeric
            assert isinstance(reason["value"], (int, float))


def test_deterministic_across_runs(tiny_package):
    package, X, _ = tiny_package
    first = Explainer(package).explain(X.head(5), ["PortScan"] * 5)
    second = Explainer(package).explain(X.head(5), ["PortScan"] * 5)
    assert first == second


def test_matches_agreed_top_reasons_shape(tiny_package):
    package, X, _ = tiny_package
    out = Explainer(package).explain(X.head(1), ["PortScan"])[0]
    assert set(out.keys()) == {"model_version", "method", "class", "reasons"}
    for r in out["reasons"]:
        assert set(r.keys()) == {"feature", "label", "value", "shap", "direction"}


def test_no_explanation_shape_when_class_unknown(tiny_package):
    """UC-2 'no explanation available' JSON shape when predicted class not in classes_."""
    package, X, _ = tiny_package
    out = Explainer(package).explain(X.head(1), ["SSH-Patator"])[0]
    assert out["method"] is None
    assert out["reasons"] == []
    assert out["note"] == NO_EXPLANATION_NOTE
    assert out["class"] == "SSH-Patator"
    assert out["model_version"] == "test-tiny-v0"


def test_empty_flows_returns_empty_list(tiny_package):
    package, X, _ = tiny_package
    assert Explainer(package).explain(X.head(0), []) == []

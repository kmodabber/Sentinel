"""TC-05: the sensor loads a valid model package and rejects a wrong one.

The package is built at test time because .gitignore blocks *.joblib.
"""
import json

import numpy as np
import pandas as pd
import pytest

from sensor.package import PACKAGE_FILES, PackageError, load_package, score
from sentinel_features.features import FEATURES, LABELS
from sentinel_ml.dummy_package import build_dummy_package


def fake_flows(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)


def edit_features(folder, change):
    """Rewrite features.json after passing its name list through change()."""
    path = folder / "features.json"
    content = json.loads(path.read_text())
    names = change([f["name"] for f in content["features"]])
    content["features"] = [{"name": n, "type": "number"} for n in names]
    path.write_text(json.dumps(content))


def test_package_has_every_c1_file(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    for name in PACKAGE_FILES:
        assert (folder / name).exists()


def test_valid_package_loads_and_scores_each_flow(tmp_path):
    package = load_package(build_dummy_package(tmp_path / "pkg"))
    flows = fake_flows()
    out = score(package, flows)
    assert len(out) == len(flows)
    assert list(out.columns) == ["predicted_class", "confidence", "model_version"]
    assert set(out["predicted_class"]) <= set(LABELS)
    assert out["confidence"].between(0, 1).all()
    assert (out["model_version"] == "sentinel-dummy-v0").all()


def test_package_with_a_missing_feature_is_refused(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    edit_features(folder, lambda names: names[:-1])
    with pytest.raises(PackageError, match="missing"):
        load_package(folder)


def test_package_with_features_out_of_order_is_refused(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    edit_features(folder, lambda names: [names[1], names[0], *names[2:]])
    with pytest.raises(PackageError, match="different order"):
        load_package(folder)


def test_package_with_a_duplicated_feature_is_refused(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    edit_features(folder, lambda names: [*names, names[0]])
    with pytest.raises(PackageError, match="twice"):
        load_package(folder)


def test_broken_features_file_is_refused(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    (folder / "features.json").write_text("{not json")
    with pytest.raises(PackageError, match="C1 format"):
        load_package(folder)


def test_package_missing_a_file_is_refused(tmp_path):
    folder = build_dummy_package(tmp_path / "pkg")
    (folder / "labels.json").unlink()
    with pytest.raises(PackageError, match="labels.json"):
        load_package(folder)


def test_features_match_the_data_card_list():
    """Guards the one shared list: 56 features, no identifiers or labels."""
    assert len(FEATURES) == 56 == len(set(FEATURES))
    assert not {"src_ip", "dst_ip", "src_port", "label", "day"} & set(FEATURES)

"""TC-04: a mismatch between the sensor and the model package stops the sensor."""

import json

import nfstream
import pytest

from sensor.package import PackageError, load_package
from sentinel_features.features import FEATURES
from sentinel_ml.dummy_package import build_dummy_package


@pytest.fixture
def pkg(tmp_path):
  """A fresh, matching dummy package in a temp folder (nothing is committed)."""
  return build_dummy_package(tmp_path / "pkg")


def edit_json(path, change):
  """Load a JSON file, apply change(data) in place, and write it back."""
  data = json.loads(path.read_text())
  change(data)
  path.write_text(json.dumps(data))


# --- matching package ---

def test_matching_package_loads(pkg):
  package = load_package(pkg)
  assert package["version"] == "sentinel-dummy-v0"
  assert package["features"] == FEATURES


# --- NFStream settings ---

def test_changed_setting_stops_sensor(pkg):
  edit_json(pkg / "nfstream_settings.json", lambda s: s.update(idle_timeout=120))
  with pytest.raises(PackageError, match="idle_timeout"):
    load_package(pkg)


def test_missing_setting_stops_sensor(pkg):
  edit_json(pkg / "nfstream_settings.json", lambda s: s.pop("n_meters"))
  with pytest.raises(PackageError, match="n_meters"):
    load_package(pkg)


def test_extra_setting_stops_sensor(pkg):
  edit_json(pkg / "nfstream_settings.json", lambda s: s.update(unknown_option=1))
  with pytest.raises(PackageError, match="unknown_option"):
    load_package(pkg)


# --- nfstream version ---

def test_wrong_nfstream_version_stops_sensor(pkg, monkeypatch):
  monkeypatch.setattr(nfstream, "__version__", "6.5.0")
  with pytest.raises(PackageError, match="6.5.0"):
    load_package(pkg)


# --- feature columns ---

def test_missing_feature_stops_sensor(pkg):
  edit_json(pkg / "features.json", lambda d: d["features"].pop())
  with pytest.raises(PackageError, match="missing"):
    load_package(pkg)


def test_swapped_feature_order_stops_sensor(pkg):
  def swap(d):
    d["features"][0], d["features"][1] = d["features"][1], d["features"][0]
  edit_json(pkg / "features.json", swap)
  with pytest.raises(PackageError, match="different order"):
    load_package(pkg)
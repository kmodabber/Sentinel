"""Load a model package (contract C1) and score flows with it.

The sensor refuses to start if the package's features.json doesn't match
the feature list in sentinel_features, so live scores never come from
columns the model wasn't trained on (SEN-FR-15, TC-05).
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sentinel_features.features import FEATURES

PACKAGE_FILES = [
    "model.joblib", "preprocess.joblib", "features.json", "labels.json",
    "nfstream_settings.json", "thresholds.json", "metrics.json", "MODEL_CARD.md",
]


class PackageError(Exception):
    """The model package is missing files or doesn't match the sensor."""


def check_features(package_features, expected=FEATURES):
    """Raise PackageError explaining how the two feature lists differ."""
    if package_features == expected:
        return
    missing = [f for f in expected if f not in package_features]
    extra = [f for f in package_features if f not in expected]
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"unexpected {extra}")
    if not problems and len(package_features) != len(expected):
        problems.append("some features are listed twice")
    if not problems:
        problems.append("same features but in a different order")
    raise PackageError("features.json doesn't match the sensor: " + "; ".join(problems))


def load_package(path):
    """Load every file in the package folder into a dict.

    Raises PackageError if a file is missing or the features don't match.
    """
    path = Path(path)
    missing = [name for name in PACKAGE_FILES if not (path / name).exists()]
    if missing:
        raise PackageError(f"model package {path} is missing {missing}")

    try:
        features = json.loads((path / "features.json").read_text())
        names = [f["name"] for f in features["features"]]
        version = features["version"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise PackageError(f"features.json in {path} isn't in the C1 format: {e!r}") from e
    check_features(names)

    return {
        "version": version,
        "features": FEATURES,
        "model": joblib.load(path / "model.joblib"),
        "preprocess": joblib.load(path / "preprocess.joblib"),
        "labels": json.loads((path / "labels.json").read_text()),
        "thresholds": json.loads((path / "thresholds.json").read_text()),
        "nfstream_settings": json.loads((path / "nfstream_settings.json").read_text()),
    }


def score(package, flows):
    """Predict a class and confidence for each flow.

    Returns one row per flow with the detection columns from contract C2:
    predicted_class, confidence and model_version. Confidence is the
    model's highest class probability.
    """
    X = package["preprocess"].transform(flows[package["features"]])
    proba = package["model"].predict_proba(X)
    classes = package["model"].classes_
    return pd.DataFrame({
        "predicted_class": classes[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1),
        "model_version": package["version"],
    }, index=flows.index)

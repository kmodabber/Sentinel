"""SEN-FR-15: a dummy model package in the real C1 format.

The sensor (SEN-74) can be built and tested against this before the real
model exists. Run from the repo root:
    python -m sentinel_ml.dummy_package

Writes models/sentinel-dummy-v0/ (git-ignored). The folder has every file
contract C1 lists; the model itself just guesses from the class mix it was
given, so its predictions mean nothing.
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer

from sentinel_features.features import ATTACK_LABELS, FEATURES, LABELS
from sentinel_features.nfstream_settings import SETTINGS

VERSION = "sentinel-dummy-v0"
OUT_DIR = Path("models") / VERSION
DEFAULT_THRESHOLD = 0.80  # same as alert.min_confidence in the sensor config (C3)

MODEL_CARD = """# {version}

**Dummy model. Do not use for real detections.**

- Built: {date}
- Data version: nfs2023te-v1 (feature list only; not trained on real data)
- Model: sklearn DummyClassifier (stratified), fitted on random rows
- Purpose: lets the sensor (SEN-74) run before the real package (SEN-68)
- Known weaknesses: predictions are random
"""


def build_model(seed=0):
    """Fit a DummyClassifier on random rows with the real feature columns.

    Every class appears at least once, so predict_proba always has a column
    for each label in LABELS.
    """
    rng = np.random.default_rng(seed)
    n = len(LABELS) * 5
    X = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    y = np.resize(LABELS, n)
    return DummyClassifier(strategy="stratified", random_state=seed).fit(X, y)


def build_dummy_package(out_dir=OUT_DIR, version=VERSION):
    """Write the eight C1 files to out_dir and return the folder path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(build_model(), out_dir / "model.joblib")
    # No scaling needed for the dummy, so preprocessing passes data through
    joblib.dump(FunctionTransformer(), out_dir / "preprocess.joblib")

    files = {
        "features.json": {
            "version": version,
            "features": [{"name": f, "type": "number"} for f in FEATURES],
        },
        "labels.json": {"classes": LABELS, "attack": ATTACK_LABELS},
        "nfstream_settings.json": SETTINGS,
        "thresholds.json": {label: DEFAULT_THRESHOLD for label in ATTACK_LABELS},
        "metrics.json": {"dummy": True, "note": "no test metrics, not a real model"},
    }
    for name, content in files.items():
        (out_dir / name).write_text(json.dumps(content, indent=2))

    (out_dir / "MODEL_CARD.md").write_text(
        MODEL_CARD.format(version=version, date=date.today().isoformat())
    )
    return out_dir


if __name__ == "__main__":
    print(f"Saved {build_dummy_package()}")

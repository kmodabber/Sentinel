"""Thin estimator wrappers kept in their own module so joblib pickles
reference a stable import path regardless of how the trainer is invoked.
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


def drop_dst_port(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the dst_port column; kept here so joblib pickles have a stable path."""
    return df.drop(columns=["dst_port"])


class StringClassLGBM(BaseEstimator, ClassifierMixin):
    """LightGBM wrapper that fits on int-encoded labels and serves string classes.

    LightGBM 4.7.0 on Windows segfaults in C-level `set_label` when given
    string or object-dtype labels; it only accepts int/float. This wrapper
    label-encodes y before fit, then re-exposes the string classes via
    `classes_` and translates predictions back. The sensor's score() uses
    only `predict_proba` and `classes_`, which this class provides.

    Kept here (not in dev_model) so joblib pickles reference the stable
    `sentinel_ml.wrappers.StringClassLGBM` path even when dev_model is run
    as a script.
    """

    def __init__(self, base=None):
        self.base = base
        self._encoder = None

    @property
    def classes_(self):
        return self._encoder.classes_ if self._encoder is not None else None

    def fit(self, X, y):
        self._encoder = LabelEncoder().fit(np.asarray(y))
        y_enc = self._encoder.transform(np.asarray(y)).astype(np.int32)
        self.base.fit(X, y_enc)
        return self

    def predict_proba(self, X):
        # LightGBM returns probability columns in sorted-label order
        # (0..n-1), which matches LabelEncoder.classes_ order, so no
        # column reordering is needed.
        return self.base.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    @property
    def feature_importances_(self):
        return self.base.feature_importances_

    @property
    def booster_(self):
        return self.base.booster_

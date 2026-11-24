"""SEN-69: per-alert SHAP explanations stored with each detection.

Uses LightGBM's built-in TreeSHAP (`booster.predict(X, pred_contrib=True)`)
which is exact and ships with LightGBM — no new dependency.

For each flow the Explainer returns a JSON dict in the shape agreed with
Humayl (see docs/eval/sen-69-top-reasons-format.md). One stored JSON feeds
GET /alerts/{id}, the Grafana alert detail, and the Discord top-reasons
line (TC-12), so the same structure must suit all three.
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

from typing import Iterable

import numpy as np
import pandas as pd

from sentinel_ml.feature_labels import label_for

TOP_K = 5
NO_EXPLANATION_NOTE = "no explanation available"


def _booster_and_classes(model):
    """Return (lightgbm Booster, classes_) for either a raw LGBMClassifier
    or the dev_model.StringClassLGBM wrapper."""
    base = getattr(model, "base", model)
    booster = getattr(base, "booster_", None)
    if booster is None:
        raise TypeError(
            f"model {type(model).__name__} has no booster_; "
            "explain.py only supports LightGBM-backed models for now"
        )
    return booster, np.asarray(model.classes_)


def _no_explanation(model_version: str, predicted_class: str) -> dict:
    return {
        "model_version": model_version,
        "method": None,
        "class": predicted_class,
        "reasons": [],
        "note": NO_EXPLANATION_NOTE,
    }


class Explainer:
    """Reusable TreeSHAP explainer.

    Build once at sensor start (SEN-NFR-01 wants median flow-end-to-alert
    <= 10s; pre-building the explainer keeps the hot path short), then call
    `explain(flows, predicted_classes)` per batch of alerts.

    Parameters
    ----------
    package : dict
        The dict returned by sensor.package.load_package(). Must include
        "model", "preprocess" and "version".
    """

    def __init__(self, package: dict):
        self.model_version = package["version"]
        self.preprocess = package["preprocess"]
        self.booster, self.classes = _booster_and_classes(package["model"])
        self.feature_names = list(self.booster.feature_name())
        self.n_features = len(self.feature_names)
        self.n_classes = len(self.classes)
        self._class_index = {cls: i for i, cls in enumerate(self.classes)}

    def explain(
        self,
        flows: pd.DataFrame,
        predicted_classes: Iterable[str],
        top_k: int = TOP_K,
    ) -> list[dict]:
        """Return one top_reasons dict per row in flows.

        `flows` is the full-feature frame (same shape the sensor passes to
        score()). `predicted_classes` is the per-row predicted label (from
        sensor.package.score). Runs preprocess internally so the same
        transformation applies to training and explanation.
        """
        predicted_classes = list(predicted_classes)
        if len(predicted_classes) != len(flows):
            raise ValueError(
                f"predicted_classes has {len(predicted_classes)} items but "
                f"flows has {len(flows)} rows"
            )
        if len(flows) == 0:
            return []

        X = self.preprocess.transform(flows)
        # booster.predict needs a plain array; a DataFrame also works but
        # casting to numpy avoids any pandas-column-reorder surprises.
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

        # TreeSHAP values: multi-class shape is (n, n_classes * (n_features+1)).
        # Binary shape is (n, n_features+1). The last column of each class
        # block is the bias; drop it.
        contrib = self.booster.predict(X_arr, pred_contrib=True)
        contrib = np.asarray(contrib)

        out = []
        for i, cls in enumerate(predicted_classes):
            cls_idx = self._class_index.get(cls)
            if cls_idx is None:
                out.append(_no_explanation(self.model_version, cls))
                continue
            if self.n_classes == 2 and contrib.shape[1] == self.n_features + 1:
                # Binary objective: a single feature contribution block.
                row = contrib[i, : self.n_features]
            else:
                start = cls_idx * (self.n_features + 1)
                row = contrib[i, start : start + self.n_features]

            order = np.argsort(np.abs(row))[::-1][:top_k]
            reasons = []
            for j in order:
                feature = self.feature_names[j]
                shap_val = float(row[j])
                raw_value = float(X_arr[i, j]) if np.issubdtype(X_arr.dtype, np.number) else X_arr[i, j]
                reasons.append({
                    "feature": feature,
                    "label": label_for(feature),
                    "value": raw_value,
                    "shap": round(shap_val, 6),
                    "direction": "up" if shap_val >= 0 else "down",
                })
            out.append({
                "model_version": self.model_version,
                "method": "tree_shap",
                "class": cls,
                "reasons": reasons,
            })
        return out


def explain(package: dict, flows: pd.DataFrame, predicted_classes: Iterable[str],
            top_k: int = TOP_K) -> list[dict]:
    """One-shot helper. Prefer the Explainer class in the sensor hot path."""
    return Explainer(package).explain(flows, predicted_classes, top_k=top_k)

"""SEN-134 tests: threshold sweep metrics stay in range and monotonic in alerts."""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

import numpy as np  # noqa: I001
import pandas as pd

from sentinel_features.features import ATTACK_LABELS, BENIGN
from sentinel_ml.threshold_sweep import (
    _metrics_at,
    recommend_global,
    recommend_per_class,
    sweep,
)


def _rows_with_per_class(per_class_recall_by_threshold: dict[float, dict[str, float | None]]) -> list[dict]:
    """Build a minimal rows list keyed by threshold with the per-class recall
    curves we need. attack_recall/benign_fpr are set to pass the overall
    targets at every threshold so the per-class behaviour is what we test.
    """
    rows = []
    for t in sorted(per_class_recall_by_threshold.keys(), reverse=True):
        rows.append({
            "threshold": round(float(t), 4),
            "attack_recall": 0.99,
            "benign_fpr": 0.0001,
            "alerts_per_10k_benign": 1.0,
            "per_class_recall": per_class_recall_by_threshold[t],
        })
    return rows


def _tiny_detections(n_benign: int = 80, n_attack: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a small detections frame that exercises the alert rule.

    Benign flows mostly predict BENIGN with low confidence; attack flows
    mostly predict an attack class with high confidence. The mix is just
    varied enough to make per-threshold metrics interesting.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_benign):
        # 80% benign, 20% false-positive attack prediction with various confidences
        if rng.random() < 0.80:
            rows.append({
                "label": BENIGN,
                "predicted_class": BENIGN,
                "confidence": float(rng.uniform(0.40, 0.95)),
            })
        else:
            rows.append({
                "label": BENIGN,
                "predicted_class": "PortScan",
                "confidence": float(rng.uniform(0.60, 0.99)),
            })
    for _ in range(n_attack):
        rows.append({
            "label": "PortScan",
            "predicted_class": "PortScan",
            "confidence": float(rng.uniform(0.55, 0.99)),
        })
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_metrics_in_range_and_shape():
    """recall/FPR stay in [0,1]; per-class recall is a dict keyed by every attack label."""
    det = _tiny_detections()
    labels = det["label"].to_numpy()
    preds = det["predicted_class"].to_numpy()
    conf = det["confidence"].to_numpy()
    for t in [0.50, 0.70, 0.80, 0.95, 0.99]:
        m = _metrics_at(labels, preds, conf, t, ATTACK_LABELS)
        assert 0.0 <= m["attack_recall"] <= 1.0, m
        assert 0.0 <= m["benign_fpr"] <= 1.0, m
        assert set(m["per_class_recall"].keys()) == set(ATTACK_LABELS)
        for cls, v in m["per_class_recall"].items():
            assert v is None or 0.0 <= v <= 1.0, (cls, v)


def test_alerts_monotonic_in_threshold():
    """TC-07: as the threshold rises, the alert count never goes up."""
    det = _tiny_detections()
    rows = sweep(det)
    last = None
    for r in rows:
        if last is not None:
            assert r["alerts_per_10k_benign"] <= last + 1e-9, (r["threshold"], r["alerts_per_10k_benign"], last)
        last = r["alerts_per_10k_benign"]


def test_recommend_global_picks_highest_feasible():
    """If several thresholds meet the targets, recommend the highest of them."""
    det = _tiny_detections(n_benign=200, n_attack=100, seed=1)
    rows = sweep(det)
    # Disable the per-class floor so this test stays about the overall targets.
    rec = recommend_global(rows, recall_target=0.0, fpr_target=1.0, class_floor=0.0)
    assert rec["meets_targets"] is True
    assert rec["threshold"] == max(r["threshold"] for r in rows)


def test_recommend_global_falls_back_when_infeasible():
    """If no threshold meets both targets, recommend the closest, flagged."""
    det = _tiny_detections()
    rows = sweep(det)
    rec = recommend_global(rows, recall_target=1.0, fpr_target=0.0)  # impossible
    assert rec["meets_targets"] is False
    assert "threshold" in rec


def test_per_class_picks_highest_threshold_at_or_below_global():
    """Class reaches recall_floor at 0.9 and 0.5, not at 0.98. With global_t=0.98
    and floor=0.80 it should get 0.9 (least loosening), NOT 0.5.
    """
    rows = _rows_with_per_class({
        0.98: {"Bot": 0.95, "DoS slowloris": 0.30},
        0.90: {"Bot": 0.95, "DoS slowloris": 0.85},
        0.50: {"Bot": 0.95, "DoS slowloris": 0.95},
    })
    out = recommend_per_class(rows, global_t=0.98, recall_floor=0.80)
    assert out["thresholds"]["Bot"] == 0.98          # already above floor
    assert out["thresholds"]["DoS slowloris"] == 0.90  # highest <= 0.98 at floor
    assert out["never_reach_floor"] == []


def test_per_class_class_already_above_floor_keeps_global_t():
    rows = _rows_with_per_class({
        0.98: {"Bot": 0.95, "DDoS": 0.99},
        0.90: {"Bot": 0.95, "DDoS": 0.99},
        0.50: {"Bot": 0.95, "DDoS": 0.99},
    })
    out = recommend_per_class(rows, global_t=0.98, recall_floor=0.80)
    assert out["thresholds"] == {"Bot": 0.98, "DDoS": 0.98}
    assert out["never_reach_floor"] == []


def test_per_class_never_reaches_floor_is_listed_and_kept_on_global():
    """A class that never reaches the floor anywhere in the sweep must stay on
    global_t and be surfaced in never_reach_floor (not silently lowered to the
    sweep minimum)."""
    rows = _rows_with_per_class({
        0.98: {"Rare attack": 0.0},
        0.90: {"Rare attack": 0.0},
        0.50: {"Rare attack": 0.05},
    })
    out = recommend_per_class(rows, global_t=0.98, recall_floor=0.80)
    assert out["thresholds"]["Rare attack"] == 0.98
    assert out["never_reach_floor"] == ["Rare attack"]


def test_recommend_global_respects_class_floor():
    """At 0.98 DoS slowloris is below class_floor=0.80 but it DOES reach 0.80
    at 0.90, so the global recommendation must drop to 0.90 (not stay at 0.98).
    Rare attack never reaches 0.80, so it's excluded and reported.
    """
    rows = _rows_with_per_class({
        0.98: {"Bot": 0.95, "DoS slowloris": 0.30, "Rare attack": 0.0},
        0.90: {"Bot": 0.95, "DoS slowloris": 0.85, "Rare attack": 0.0},
        0.50: {"Bot": 0.95, "DoS slowloris": 0.95, "Rare attack": 0.05},
    })
    rec = recommend_global(rows, recall_target=0.90, fpr_target=0.01, class_floor=0.80)
    assert rec["meets_targets"] is True
    assert rec["threshold"] == 0.90
    assert "Rare attack" in rec["never_reach_class_floor"]
    assert "DoS slowloris" not in rec["never_reach_class_floor"]


def test_recommend_global_class_floor_zero_matches_old_behaviour():
    """With class_floor=0 the picker is purely overall-target-driven."""
    rows = _rows_with_per_class({
        0.98: {"Bot": 0.95, "DoS slowloris": 0.30},
        0.90: {"Bot": 0.95, "DoS slowloris": 0.85},
        0.50: {"Bot": 0.95, "DoS slowloris": 0.95},
    })
    rec = recommend_global(rows, recall_target=0.90, fpr_target=0.01, class_floor=0.0)
    assert rec["threshold"] == 0.98


def test_empty_benign_returns_nan_fpr():
    """Guard against divide-by-zero when a slice has no benign rows."""
    det = pd.DataFrame([
        {"label": "PortScan", "predicted_class": "PortScan", "confidence": 0.9},
        {"label": "PortScan", "predicted_class": "PortScan", "confidence": 0.8},
    ])
    m = _metrics_at(det["label"].to_numpy(), det["predicted_class"].to_numpy(),
                    det["confidence"].to_numpy(), 0.7, ATTACK_LABELS)
    assert m["benign_fpr"] != m["benign_fpr"]  # NaN check
    assert m["attack_recall"] == 1.0

"""SEN-135 unit tests: slice-label mapping + verdict rules.

Only exercises pure helpers — no nfstream, no PCAP, no model package.
"""

from __future__ import annotations

import json

from demo.check_slices import (
    HARDCODED_RARE_MERGED,
    RARE_LABEL,
    _load_rare_merged,
    _verdict,
    map_slice_label_to_model_class,
)
from sentinel_features.features import LABELS


def test_direct_label_maps_to_itself():
    model_class, status = map_slice_label_to_model_class(
        "PortScan", rare_merged=set(), labels=LABELS
    )
    assert model_class == "PortScan"
    assert status == "direct"


def test_rare_merged_label_maps_to_rare_attack():
    model_class, status = map_slice_label_to_model_class(
        "Web Attack - Sql Injection",
        rare_merged={"Web Attack - Sql Injection", "Web Attack - XSS"},
        labels=LABELS,
    )
    assert model_class == RARE_LABEL
    assert status == "rare"


def test_unknown_label_passes_through_with_status():
    model_class, status = map_slice_label_to_model_class(
        "Totally Not A Class", rare_merged={"Web Attack - XSS"}, labels=LABELS
    )
    assert model_class == "Totally Not A Class"
    assert status == "unknown"


def test_load_rare_merged_reads_data_card(tmp_path):
    card = tmp_path / "data_card.json"
    card.write_text(json.dumps({
        "rare_classes": {"merged": ["Heartbleed", "Infiltration"]},
    }))
    merged = _load_rare_merged(card)
    assert merged == {"Heartbleed", "Infiltration"}


def test_load_rare_merged_falls_back_when_card_missing(tmp_path):
    merged = _load_rare_merged(tmp_path / "does_not_exist.json")
    assert merged == HARDCODED_RARE_MERGED


def test_verdict_hit_requires_direct_or_rare_mapping():
    # Direct mapping, enough alerts, enough hit-class → HIT.
    assert _verdict(0.80, 0.90, "PortScan", mapping_status="direct") == "HIT"


def test_verdict_miss_on_low_alert_share():
    assert _verdict(0.05, 0.90, "PortScan", mapping_status="direct") == "MISS"


def test_verdict_unknown_mapping_never_returns_hit():
    """Even with lots of alerts and 'matching' class share, an unknown mapping
    must NOT silently be called a HIT — the whole premise is undefined."""
    v = _verdict(0.90, 0.90, "Totally Not A Class", mapping_status="unknown")
    assert v in {"PARTIAL", "MISS"}
    assert v != "HIT"


def test_verdict_partial_when_class_share_too_low():
    assert _verdict(0.80, 0.30, "PortScan", mapping_status="direct") == "PARTIAL"

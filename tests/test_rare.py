"""SEN-FR-05: rare classes are merged into Rare attack on purpose."""
import pandas as pd

from sentinel_data.clean import feature_columns
from sentinel_data.split import (
    RARE_LABEL,
    data_card,
    find_rare_classes,
    merge_rare_classes,
    split,
)


def fixture_frame():
    """BENIGN and PortScan are common, Heartbleed and XSS are rare (under 100)."""
    rows = []
    for label, n in [("BENIGN", 500), ("PortScan", 300), ("Heartbleed", 11), ("Web Attack - XSS", 27)]:
        for i in range(n):
            rows.append({
                "day": "Thursday",
                "label": label,
                "bidirectional_first_seen_ms": i * 10_000,
                "bidirectional_duration_ms": 5,
                "bidirectional_packets": i % 7,
            })
    return pd.DataFrame(rows)


def test_finds_only_small_attack_classes():
    rare = find_rare_classes(fixture_frame(), min_flows=100)
    assert rare == ["Heartbleed", "Web Attack - XSS"]


def test_benign_is_never_rare():
    df = fixture_frame()
    df = df[(df["label"] != "BENIGN") | (df.index < 5)]
    assert "BENIGN" not in find_rare_classes(df, min_flows=100)


def test_rare_merged_common_untouched():
    df = fixture_frame()
    out = merge_rare_classes(df, ["Heartbleed", "Web Attack - XSS"])
    assert set(out["label"]) == {"BENIGN", "PortScan", RARE_LABEL}
    assert (out["label"] == RARE_LABEL).sum() == 11 + 27
    # common classes keep their own label
    assert (out["label"] == "PortScan").sum() == 300


def test_original_label_kept_and_not_a_feature():
    df = fixture_frame()
    out = merge_rare_classes(df, ["Heartbleed"])
    assert (out["label_original"] == df["label"]).all()
    assert "label_original" not in feature_columns(out)


def test_data_card_lists_merged_classes():
    df = fixture_frame()
    rare = find_rare_classes(df, min_flows=100)
    out = merge_rare_classes(split(df, gap_seconds=0), rare)
    card = data_card(out, len(df), 0, rare, 100)
    rc = card["rare_classes"]
    assert rc["merged"] == ["Heartbleed", "Web Attack - XSS"]
    assert rc["min_flows"] == 100
    assert set(rc["rows_per_original_class"]) == {"Heartbleed", "Web Attack - XSS"}
    assert RARE_LABEL in card["rows_per_class"]
    assert "Heartbleed" not in card["rows_per_class"]
    # every rare attack still shows up in each part, split on its own window
    for counts in rc["rows_per_original_class"].values():
        assert all(counts[s] > 0 for s in ("train", "val", "test"))


def test_no_rare_classes_means_no_change():
    df = fixture_frame()
    out = merge_rare_classes(df, [])
    assert (out["label"] == df["label"]).all()

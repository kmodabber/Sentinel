"""SEN-FR-05: rare classes are merged into Rare attack on purpose."""
import pandas as pd

from sentinel_data.clean import feature_columns
from sentinel_data.split import (
    RARE_LABEL,
    RARE_MIN_FLOWS,
    data_card,
    find_rare_classes,
    merge_rare_classes,
    split,
)
from sentinel_features.features import LABELS as CANONICAL_LABELS


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


# --- Contract: RARE_MIN_FLOWS threshold and sentinel_features.LABELS cannot
# drift apart. If someone changes the threshold or the canonical list, this
# test is what should catch it. Counts match the real data v1 clean.parquet.
DATA_V1_CLEAN_COUNTS = {
    "BENIGN": 2628851,
    "Bot": 737,
    "DDoS": 95443,
    "DoS GoldenEye": 7504,
    "DoS Hulk": 156498,
    "DoS Slowhttptest": 2584,
    "DoS slowloris": 4871,
    "FTP-Patator": 7940,
    "Heartbleed": 9,
    "Infiltration": 28,
    "PortScan": 230624,
    "SSH-Patator": 2960,
    "Web Attack - Brute Force": 147,  # above RARE_MIN_FLOWS=100 -> its own class
    "Web Attack - Sql Injection": 12,
    "Web Attack - XSS": 25,
}


def _data_v1_shaped_frame():
    """One row per (label, i) with just the fields find_rare_classes needs."""
    rows = []
    for label, n in DATA_V1_CLEAN_COUNTS.items():
        rows.extend({"label": label} for _ in range(n))
    return pd.DataFrame(rows)


def test_threshold_and_canonical_labels_agree():
    """The RARE_MIN_FLOWS=100 threshold applied to data v1 counts produces
    exactly the 12 classes in sentinel_features.features.LABELS. If this
    fails, split.py and the canonical contract have drifted apart."""
    df = _data_v1_shaped_frame()
    rare = find_rare_classes(df, min_flows=RARE_MIN_FLOWS)
    merged = merge_rare_classes(df, rare)
    assert set(merged["label"].unique()) == set(CANONICAL_LABELS), (
        f"threshold-derived classes {sorted(set(merged['label'].unique()))} "
        f"!= canonical {sorted(CANONICAL_LABELS)}"
    )


def test_web_attack_brute_force_is_above_threshold():
    """Explicit check against the 'off by one at 100' question raised in review:
    Web Attack - Brute Force has 147 flows on the full clean data, so it stays
    as its own class and is not merged into Rare attack."""
    df = _data_v1_shaped_frame()
    assert (df["label"] == "Web Attack - Brute Force").sum() == 147
    rare = find_rare_classes(df, min_flows=RARE_MIN_FLOWS)
    assert "Web Attack - Brute Force" not in rare

"""TC-02: cleaning removes bad values (SEN-FR-03)."""
import numpy as np
import pandas as pd

from sentinel_data.clean import ID_COLS, clean, feature_columns


def fixture_frame():
    """Small table with one example of each problem the cleaning step handles.

    row 1: NaN value          row 3: infinite value
    row 2: negative duration  row 5: copy of row 0 except for its id
    vlan_id is the same on every row, so it should be dropped as constant.
    """
    df = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6],
        "expiration_id": [-1, 0, 0, 1, -1, -1, 0],
        "src_ip": ["10.0.0.1"] * 7,
        "dst_ip": ["10.0.0.2"] * 7,
        "src_port": [5000, 5001, 5002, 5003, 5004, 5000, 5005],
        "dst_port": [80, 22, 22, 443, 80, 80, 21],
        "vlan_id": [0] * 7,
        "bidirectional_first_seen_ms": [100, 200, 300, 400, 500, 100, 600],
        "bidirectional_last_seen_ms": [110, 210, 310, 410, 510, 110, 610],
        "bidirectional_duration_ms": [10, 10, -5, 10, 20, 10, 30],
        "bidirectional_packets": [3, 4, 5, 6, 7, 3, 8],
        "bidirectional_mean_ps": [60.0, np.nan, 70.0, np.inf, 80.0, 60.0, 90.0],
        "label": ["BENIGN", "SSH-Patator", "SSH-Patator", "BENIGN", "BENIGN", "BENIGN", "FTP-Patator"],
        "day": ["Tuesday"] * 7,
    })
    return df


def test_bad_values_removed():
    """No NaN, infinite or negative-duration rows are left, and the counts are right."""
    out, s = clean(fixture_frame())
    feats = out[feature_columns(out)]
    assert not feats.isna().any().any()
    assert not np.isinf(feats.select_dtypes("number")).any().any()
    assert (out["bidirectional_duration_ms"] >= 0).all()
    assert s["dropped_nan_or_inf"] == 2
    assert s["dropped_negative_duration"] == 1


def test_exact_duplicates_dropped_ignoring_row_id():
    """Rows 0 and 5 only differ in id, so one of them is dropped."""
    out, s = clean(fixture_frame())
    assert s["dropped_duplicates"] == 1
    assert s["rows_out"] == 3 == len(out)


def test_identifiers_and_constants_not_features():
    """IPs, ids, ports and timestamps aren't features, but dst_port still is."""
    out, s = clean(fixture_frame())
    feats = feature_columns(out)
    assert not set(feats) & set(ID_COLS)
    assert "vlan_id" in s["dropped_constant_columns"]
    assert "vlan_id" not in out.columns
    assert "bidirectional_first_seen_ms" not in feats
    assert "label" not in feats and "day" not in feats
    assert "dst_port" in feats


def test_input_not_modified():
    """clean() works on a copy, the table passed in stays the same."""
    df = fixture_frame()
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)

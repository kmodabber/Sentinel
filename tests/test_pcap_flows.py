import os
from pathlib import Path

import pandas as pd
import pytest

from sensor import pcap_flows
from sensor.pcap_flows import METADATA_COLUMNS, flows_from_pcap
from sentinel_features.columns import MissingFeaturesError, select_features
from sentinel_features.features import FEATURES


def test_select_features_orders_and_drops_extras():
    df = pd.DataFrame({c: [0] for c in reversed(FEATURES)})
    df["src_ip"] = "10.0.0.1"
    assert list(select_features(df).columns) == FEATURES


def test_select_features_missing_column_raises():
    df = pd.DataFrame({c: [0] for c in FEATURES}).drop(columns=["protocol"])
    with pytest.raises(MissingFeaturesError):
        select_features(df)


class _FakeStreamer:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _fake_flows(drop=()):
    cols = list(dict.fromkeys(FEATURES + METADATA_COLUMNS))
    df = pd.DataFrame({c: [1, 2] for c in cols})
    df["src_ip"] = ["10.0.0.1", "10.0.0.2"]
    df["dst_ip"] = ["10.0.0.9", "10.0.0.9"]
    return df.drop(columns=list(drop))


@pytest.fixture
def fake_pcap(tmp_path, monkeypatch):
    path = tmp_path / "x.pcap"
    path.write_bytes(b"")

    def install(df):
        monkeypatch.setattr(
            pcap_flows, "build_streamer", lambda _p: _FakeStreamer(df)
        )
        return path

    return install


def test_default_returns_only_features(fake_pcap):
    flows = flows_from_pcap(fake_pcap(_fake_flows()))
    assert list(flows.columns) == FEATURES


def test_keep_metadata_puts_metadata_before_features(fake_pcap):
    flows = flows_from_pcap(fake_pcap(_fake_flows()), keep_metadata=True)
    extra = [c for c in METADATA_COLUMNS if c not in FEATURES]
    assert list(flows.columns) == extra + FEATURES
    assert list(flows["src_ip"]) == ["10.0.0.1", "10.0.0.2"]
    assert len(flows) == 2


def test_keep_metadata_missing_column_raises(fake_pcap):
    path = fake_pcap(_fake_flows(drop=["src_ip"]))
    with pytest.raises(KeyError, match="src_ip"):
        flows_from_pcap(path, keep_metadata=True)


TUESDAY_PCAP = os.environ.get("SENTINEL_TUESDAY_PCAP")

needs_pcap = pytest.mark.skipif(
    not TUESDAY_PCAP or not Path(TUESDAY_PCAP).is_file(),
    reason="set SENTINEL_TUESDAY_PCAP to a local PCAP slice",
)


@needs_pcap
def test_pcap_gives_56_columns_in_order():
    flows = flows_from_pcap(TUESDAY_PCAP)
    assert len(flows.columns) == 56
    assert list(flows.columns) == FEATURES
    assert len(flows) > 0


@needs_pcap
def test_pcap_keeps_metadata_columns():
    flows = flows_from_pcap(TUESDAY_PCAP, keep_metadata=True)
    assert set(METADATA_COLUMNS) <= set(flows.columns)
    assert list(flows.columns[-len(FEATURES):]) == FEATURES
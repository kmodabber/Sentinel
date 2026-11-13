import os
from pathlib import Path

import pandas as pd
import pytest

from sensor.pcap_flows import flows_from_pcap
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


TUESDAY_PCAP = os.environ.get("SENTINEL_TUESDAY_PCAP")

@pytest.mark.skipif(not TUESDAY_PCAP or not Path(TUESDAY_PCAP).is_file(), reason="set SENTINEL_TUESDAY_PCAP to a local PCAP slice")
def test_pcap_gives_56_columns_in_order():
  flows = flows_from_pcap(TUESDAY_PCAP)
  assert len(flows.columns) == 56
  assert list(flows.columns) == FEATURES
  assert len(flows) > 0
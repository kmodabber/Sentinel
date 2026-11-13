"""
Turn a PCAP file into flows with the dataset's NFStream settings.

Returns only the model's feature columns, in the model's order.
"""

from pathlib import Path

import pandas as pd

from sentinel_features.columns import select_features
from sentinel_features.nfstream_config import build_streamer


def flows_from_pcap(pcap_path) -> pd.DataFrame:
  path = Path(pcap_path)
  if not path.is_file():
    raise FileNotFoundError(f"PCAP not found: {path}")

  flows = build_streamer(str(path)).to_pandas()
  if flows is None or flows.empty:
    raise ValueError(f"no flows produced from {path}")
  return select_features(flows)
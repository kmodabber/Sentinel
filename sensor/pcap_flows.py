"""
Turn a PCAP file into flows with the dataset's NFStream settings.

Returns only the model's feature columns, in the model's order.
"""

from pathlib import Path

import pandas as pd

from sentinel_features.columns import select_features
from sentinel_features.nfstream_config import build_streamer

METADATA_COLUMNS = [
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
]


def flows_from_pcap(pcap_path, keep_metadata: bool = False) -> pd.DataFrame:
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP not found: {path}")

    flows = build_streamer(str(path)).to_pandas()
    if flows is None or flows.empty:
        raise ValueError(f"no flows produced from {path}")

    features = select_features(flows)
    if not keep_metadata:
        return features

    missing = [c for c in METADATA_COLUMNS if c not in flows.columns]
    if missing:
        raise KeyError(f"NFStream output is missing columns: {missing}")

    # Skip any metadata column that is already a model feature.
    extra = [c for c in METADATA_COLUMNS if c not in features.columns]
    return pd.concat([flows[extra], features], axis=1)

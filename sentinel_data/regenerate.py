"""Regenerate one CICIDS2017 day with NFStream and the published labeller.

The flow pipeline and the post-filter on single-packet RST/FIN flows are
adapted from the authors' notebook 3-NFS-2023-TE-1-analyse-dataset.ipynb:
https://github.com/FlowFrontiers/CyberML-DataQuality/blob/main/4-NFS-2023/3-NFS-2023-TE-1-analyse-dataset.ipynb
Copied on 2026-10-28.

Paper: "Evaluating ML-Based Anomaly Detection Across Datasets of Varied
Integrity: A Case Study".
"""

import logging
import os
import time
from datetime import timedelta

import nfstream

from sentinel_data.label_plugins import FlowLabelManager, PayloadManager
from sentinel_features.nfstream_config import build_streamer

logger = logging.getLogger(__name__)


def regenerate(day: str, pcap_path: str, output_path: str):
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(pcap_path)
    logger.info(f"NFStream version: {nfstream.__version__}")
    logger.info(f"Processing {pcap_path} into {output_path}")

    start = time.time()
    streamer = build_streamer(
        pcap_path, extra_udps=[PayloadManager(), FlowLabelManager(day)]
    )

    df = streamer.to_pandas(columns_to_anonymize=[])
    logger.info(f"NFStream generated flows: {len(df)}")
    logger.info(f"Time required: {timedelta(seconds=time.time() - start)}")

    keep = ~(
        (df["bidirectional_packets"] == 1)
        & (
            (df["bidirectional_rst_packets"] == 1)
            | (df["bidirectional_fin_packets"] == 1)
        )
    )
    out = df[keep].rename(columns={"udps.label": "label"})
    logger.info(f"Flows filtered: {len(df) - len(out)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Flows stored after filtering: {len(out)}")

# Regenerate Tuesday data from CICIDS2017
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    regenerate(
        "Tuesday",
        "PCAP/deduplicated_reordered/Tuesday-WorkingHours.pcap",
        "NFS-2023-TE/datasets/Tuesday_Regen.csv",
    )
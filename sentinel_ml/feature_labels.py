"""Plain-English labels for the 56 model features (SEN-69).

Used by sentinel_ml/explain.py to turn a SHAP top-5 into the short reasons
that go into detections.top_reasons (C2) and the Discord alert line (C5).

Labels are deliberately short, neutral and direction-agnostic. The
"direction" (up/down) comes from the SHAP sign and is stored separately
in the top_reasons JSON. So the same label applies whether a feature
pushed the prediction up or down.

If we want to put this mapping inside the model package later, that's a
C1 change and needs both ML + sensor leads to sign off.
"""

from __future__ import annotations

from sentinel_features.features import FEATURES

FEATURE_LABELS: dict[str, str] = {
    # Identity-ish
    "dst_port": "destination port",
    "protocol": "IP protocol number",

    # Durations
    "bidirectional_duration_ms": "flow duration",
    "src2dst_duration_ms": "client-to-server duration",
    "dst2src_duration_ms": "server-to-client duration",

    # Packet / byte counts
    "bidirectional_packets": "total packets",
    "bidirectional_bytes": "total bytes",
    "src2dst_packets": "client-to-server packets",
    "src2dst_bytes": "client-to-server bytes",
    "dst2src_packets": "server-to-client packets",
    "dst2src_bytes": "server-to-client bytes",

    # Packet-size stats (bidirectional)
    "bidirectional_min_ps": "smallest packet size",
    "bidirectional_mean_ps": "mean packet size",
    "bidirectional_stddev_ps": "packet-size variability",
    "bidirectional_max_ps": "largest packet size",

    # Packet-size stats (src2dst)
    "src2dst_min_ps": "smallest client-to-server packet",
    "src2dst_mean_ps": "mean client-to-server packet size",
    "src2dst_stddev_ps": "client-to-server packet-size variability",
    "src2dst_max_ps": "largest client-to-server packet",

    # Packet-size stats (dst2src)
    "dst2src_min_ps": "smallest server-to-client packet",
    "dst2src_mean_ps": "mean server-to-client packet size",
    "dst2src_stddev_ps": "server-to-client packet-size variability",
    "dst2src_max_ps": "largest server-to-client packet",

    # Inter-arrival times (bidirectional)
    "bidirectional_min_piat_ms": "shortest gap between packets",
    "bidirectional_mean_piat_ms": "mean gap between packets",
    "bidirectional_stddev_piat_ms": "inter-packet-gap variability",
    "bidirectional_max_piat_ms": "longest gap between packets",

    # Inter-arrival times (src2dst)
    "src2dst_min_piat_ms": "shortest client-to-server gap",
    "src2dst_mean_piat_ms": "mean client-to-server gap",
    "src2dst_stddev_piat_ms": "client-to-server gap variability",
    "src2dst_max_piat_ms": "longest client-to-server gap",

    # Inter-arrival times (dst2src)
    "dst2src_min_piat_ms": "shortest server-to-client gap",
    "dst2src_mean_piat_ms": "mean server-to-client gap",
    "dst2src_stddev_piat_ms": "server-to-client gap variability",
    "dst2src_max_piat_ms": "longest server-to-client gap",

    # TCP flag counts (bidirectional)
    "bidirectional_syn_packets": "SYN packet count",
    "bidirectional_cwr_packets": "CWR (congestion window) flag count",
    "bidirectional_ece_packets": "ECE (ECN echo) flag count",
    "bidirectional_ack_packets": "ACK packet count",
    "bidirectional_psh_packets": "PSH packet count",
    "bidirectional_rst_packets": "reset (RST) packet count",
    "bidirectional_fin_packets": "FIN packet count",

    # TCP flag counts (src2dst)
    "src2dst_syn_packets": "client SYN packets",
    "src2dst_cwr_packets": "client CWR flag count",
    "src2dst_ece_packets": "client ECE flag count",
    "src2dst_ack_packets": "client ACK packets",
    "src2dst_psh_packets": "client PSH packets",
    "src2dst_rst_packets": "client RST packets",
    "src2dst_fin_packets": "client FIN packets",

    # TCP flag counts (dst2src)
    "dst2src_syn_packets": "server SYN packets",
    "dst2src_cwr_packets": "server CWR flag count",
    "dst2src_ece_packets": "server ECE flag count",
    "dst2src_ack_packets": "server ACK packets",
    "dst2src_psh_packets": "server PSH packets",
    "dst2src_rst_packets": "server RST packets",
    "dst2src_fin_packets": "server FIN packets",
}


def label_for(feature: str) -> str:
    """Return the plain-English label for a feature; fall back to the raw name."""
    return FEATURE_LABELS.get(feature, feature)


# Fail fast if the mapping drifts out of sync with the feature list.
_missing = [f for f in FEATURES if f not in FEATURE_LABELS]
if _missing:
    raise RuntimeError(
        f"feature_labels.py is out of sync with FEATURES: missing {_missing}"
    )

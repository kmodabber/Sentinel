"""The model's feature list and class names, kept in one place.

FEATURES is the 56-feature list from data_card.json (data v1, nfs2023te-v1),
in the same order. The model package's features.json must match it exactly,
and the sensor refuses a package that doesn't (contract C1, TC-05).
"""

BENIGN = "BENIGN"

FEATURES = [
    "dst_port", "protocol",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps",
    "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "bidirectional_syn_packets", "bidirectional_cwr_packets",
    "bidirectional_ece_packets", "bidirectional_ack_packets",
    "bidirectional_psh_packets", "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "src2dst_syn_packets", "src2dst_cwr_packets", "src2dst_ece_packets",
    "src2dst_ack_packets", "src2dst_psh_packets", "src2dst_rst_packets",
    "src2dst_fin_packets",
    "dst2src_syn_packets", "dst2src_cwr_packets", "dst2src_ece_packets",
    "dst2src_ack_packets", "dst2src_psh_packets", "dst2src_rst_packets",
    "dst2src_fin_packets",
]

# Classes in data v1, after rare classes were merged (SEN-FR-05)
LABELS = [
    BENIGN, "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "PortScan", "Rare attack", "SSH-Patator",
    "Web Attack - Brute Force",
]

ATTACK_LABELS = [label for label in LABELS if label != BENIGN]

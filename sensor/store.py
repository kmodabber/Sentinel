"""Write scored flows to PostgreSQL (contract C2, SEN-FR-17).

Every finished flow becomes one row in flows and one row in detections
(predicted class, confidence, model version). The threshold arrives in
SEN-FR-18, so is_alert stays false for now.
"""

import json
import os
from datetime import datetime, timezone

PROTOCOLS = {1: "ICMP", 6: "TCP", 17: "UDP"}

INSERT_MODEL = """
INSERT INTO model_versions (version, metrics)
VALUES (%s, %s)
ON CONFLICT (version) DO NOTHING
"""

INSERT_FLOW = """
INSERT INTO flows (seen_at, src_ip, dst_ip, src_port, dst_port, protocol,
                   duration_ms, packets, bytes, source)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

INSERT_DETECTION = """
INSERT INTO detections (flow_id, model_version, predicted_class, confidence, is_alert)
VALUES (%s, %s, %s, %s, FALSE)
"""

NEEDED = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "bidirectional_first_seen_ms", "bidirectional_duration_ms",
    "bidirectional_packets", "bidirectional_bytes",
]


def connect(url=None):
    """Open a PostgreSQL connection from DATABASE_URL (or the given url)."""
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    import psycopg  # imported here so the rest of the sensor works without it

    return psycopg.connect(url)


def protocol_name(number):
    return PROTOCOLS.get(int(number), str(int(number)))


def flow_row(flow, source):
    """Map one NFStream flow to the columns of the flows table."""
    seen_at = datetime.fromtimestamp(
        float(flow["bidirectional_first_seen_ms"]) / 1000, tz=timezone.utc
    )
    return (
        seen_at,
        str(flow["src_ip"]),
        str(flow["dst_ip"]),
        int(flow["src_port"]),
        int(flow["dst_port"]),
        protocol_name(flow["protocol"]),
        float(flow["bidirectional_duration_ms"]),
        int(flow["bidirectional_packets"]),
        int(flow["bidirectional_bytes"]),
        source,
    )


def register_model(conn, version, metrics=None):
    """Make sure the model version exists, because detections reference it."""
    with conn.cursor() as cur:
        cur.execute(INSERT_MODEL, (version, json.dumps(metrics or {})))


def store_scored_flows(conn, flows, scores, source="replay"):
    """Insert every flow and its detection in one transaction.

    flows needs the NEEDED columns (flows_from_pcap with keep_metadata=True)
    and scores is the frame returned by sensor.package.score, row for row.
    Returns the number of flows written. On any error nothing is kept.
    """
    if len(flows) != len(scores):
        raise ValueError(f"{len(flows)} flows but {len(scores)} scores")
    missing = [c for c in NEEDED if c not in flows.columns]
    if missing:
        raise KeyError(f"flows are missing columns: {missing}")

    try:
        with conn.cursor() as cur:
            for (_, flow), (_, result) in zip(flows.iterrows(), scores.iterrows()):
                cur.execute(INSERT_FLOW, flow_row(flow, source))
                flow_id = cur.fetchone()[0]
                confidence = min(max(float(result["confidence"]), 0.0), 1.0)
                cur.execute(INSERT_DETECTION, (
                    flow_id,
                    result["model_version"],
                    str(result["predicted_class"]),
                    confidence,
                ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(flows)


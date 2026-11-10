"""SEN-FR-20 synthetic database feed for Sentinel C2 v0.

Writes to existing flows, model_versions and detections tables.
For development/test databases ONLY; never run against a live capture store.
"""
import argparse
import logging
import os
import random
import signal
import time
from datetime import datetime, timezone
from ipaddress import IPv4Address

LOG = logging.getLogger("fake_detections")
STOP = False
MODEL_VERSION = "synthetic-fr20-v1"
CLASSES = ("Port Scan", "SSH Brute Force", "DDoS", "Web Attack")
# C2's source CHECK does not permit "synthetic". Use "replay" as a
# development-only stand-in, and identify fake records by model_version.
SOURCE = "replay"


def on_stop(*_):
    global STOP
    STOP = True


def generate_event(rng, timestamp=None, attack_probability=0.25):
    if not 0 <= attack_probability <= 1:
        raise ValueError("attack_probability must be between 0 and 1")
    attack = rng.random() < attack_probability
    label = rng.choice(CLASSES) if attack else "Benign"
    port = {"SSH Brute Force": 22, "Web Attack": 443, "DDoS": 80}.get(
        label, rng.choice([22, 53, 80, 443, 8080])
    )
    src = str(IPv4Address(rng.randint(int(IPv4Address("198.18.0.1")),
                                     int(IPv4Address("198.19.255.254")))))
    dst = f"10.0.0.{rng.randint(1, 254)}"
    packets = (rng.randint(1000, 20000) if label == "DDoS"
               else rng.randint(2, 30) if label == "Port Scan"
               else rng.randint(5, 400))
    byte_count = packets * rng.randint(60, 1400)
    confidence = round(rng.uniform(0.80, 0.99), 4) if attack else round(rng.uniform(0.60, 0.99), 4)
    return {
        "seen_at": timestamp or datetime.now(timezone.utc),
        "src_ip": src, "dst_ip": dst,
        "src_port": rng.randint(1024, 65535), "dst_port": port,
        "protocol": "UDP" if port == 53 else "TCP",
        "duration_ms": round(rng.uniform(10, 20000), 2),
        "packets": packets, "bytes": byte_count,
        "source": SOURCE,
        "predicted_class": label,
        "confidence": confidence,
        "is_attack": attack,
        "is_alert": attack and confidence >= 0.85,
    }


def verify_schema(conn):
    required = {
        "flows": {"id", "seen_at", "src_ip", "dst_ip", "src_port", "dst_port",
                  "protocol", "duration_ms", "packets", "bytes", "source"},
        "model_versions": {"version", "created_at", "metrics", "active"},
        "detections": {"id", "flow_id", "model_version", "predicted_class",
                       "confidence", "is_alert", "top_reasons", "created_at"},
    }
    for table, expected in required.items():
        rows = conn.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = %s""",
            (table,),
        ).fetchall()
        missing = expected - {row[0] for row in rows}
        if missing:
            raise RuntimeError(f"C2 schema mismatch in {table}: missing {sorted(missing)}")


def register_model(conn):
    # FK target must exist before detection insert.
    conn.execute(
        """INSERT INTO model_versions(version, metrics, active)
           VALUES (%s, %s::jsonb, FALSE)
           ON CONFLICT (version) DO NOTHING""",
        (MODEL_VERSION, '{"synthetic": true, "ticket": "SEN-FR-20"}'),
    )


def insert_event(conn, event):
    """Commit flow and optional detection atomically. Return (flow_id, has_detection)."""
    with conn.transaction():
        flow_id = conn.execute(
            """INSERT INTO flows
               (seen_at, src_ip, dst_ip, src_port, dst_port, protocol,
                duration_ms, packets, bytes, source)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               RETURNING id""",
            tuple(event[key] for key in (
                "seen_at", "src_ip", "dst_ip", "src_port", "dst_port",
                "protocol", "duration_ms", "packets", "bytes", "source"
            )),
        ).fetchone()[0]
        if event["is_attack"]:
            conn.execute(
                """INSERT INTO detections
                   (flow_id, model_version, predicted_class, confidence,
                    is_alert, top_reasons)
                   VALUES (%s,%s,%s,%s,%s,%s::jsonb)""",
                (flow_id, MODEL_VERSION, event["predicted_class"],
                 event["confidence"], event["is_alert"],
                 '{"note":"synthetic data for dashboard development"}'),
            )
    return flow_id, event["is_attack"]


def run(conn, count=0, interval=1.0, seed=20, attack_probability=0.25):
    if count < 0 or interval <= 0:
        raise ValueError("count must be >= 0 and interval > 0")
    verify_schema(conn)
    register_model(conn)
    rng = random.Random(seed)
    total = detections = 0
    next_tick = time.monotonic()
    while not STOP and (count == 0 or total < count):
        event = generate_event(rng, attack_probability=attack_probability)
        flow_id, detected = insert_event(conn, event)
        total += 1
        detections += int(detected)
        LOG.info("flow=%s label=%s confidence=%.3f total=%s detections=%s",
                 flow_id, event["predicted_class"], event["confidence"],
                 total, detections)
        next_tick += interval
        while not STOP and time.monotonic() < next_tick:
            time.sleep(min(0.1, next_tick - time.monotonic()))
    return total, detections


def main():
    parser = argparse.ArgumentParser(description="SEN-FR-20 C2 synthetic database feed")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of flows; 0 means continuous")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between flows")
    parser.add_argument("--attack-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    if args.count < 0 or args.interval <= 0 or not 0 <= args.attack_probability <= 1:
        parser.error("count >= 0, interval > 0 and attack-probability in [0,1] required")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("Set DATABASE_URL to your disposable Sentinel PostgreSQL database")
    import psycopg
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)
    with psycopg.connect(dsn, autocommit=True) as conn:
        count, detected = run(conn, args.count, args.interval,
                              args.seed, args.attack_probability)
    LOG.info("Completed: %d flows, %d detections", count, detected)


if __name__ == "__main__":
    main()

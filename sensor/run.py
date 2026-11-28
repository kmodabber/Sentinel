"""Run the sensor on a PCAP file: flows -> score -> PostgreSQL.

    python -m sensor.run --pcap slice.pcap --model models/sentinel-dummy-v0

Reads DATABASE_URL from the environment. From the host use localhost, e.g.
postgresql://sentinel:<password>@localhost:5432/sentinel
"""

import argparse
import json
from pathlib import Path

from sensor import store
from sensor.package import load_package, score
from sensor.pcap_flows import flows_from_pcap


def run(pcap, model_dir, conn, source="replay"):
    """Score every flow in the PCAP and store it. Returns the flow count."""
    package = load_package(model_dir)
    flows = flows_from_pcap(pcap, keep_metadata=True)
    scores = score(package, flows)

    metrics_file = Path(model_dir) / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    store.register_model(conn, package["version"], metrics)
    return store.store_scored_flows(conn, flows, scores, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--model", required=True, help="model package folder")
    args = parser.parse_args()

    conn = store.connect()
    try:
        n = run(args.pcap, args.model, conn)
    finally:
        conn.close()
    print(f"Stored {n} flows with detections")


if __name__ == "__main__":
    main()

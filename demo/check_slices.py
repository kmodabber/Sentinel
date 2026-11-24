"""SEN-135: run every demo PCAP slice through the real sensor code path.

For each slice from demo/slice.csv:
  1. Load flows with sensor.pcap_flows.flows_from_pcap(..., keep_metadata=True).
     The sensor's flow builder is the single source of truth; we don't
     reimplement NFStream or filter flows ourselves.
  2. Score them with sensor.package.score() using the supplied package.
  3. Split into attacker<->victim traffic (the ground-truth attack) and
     "other" traffic (potential false alarms).
  4. Count alerts at the SEN-134 threshold, sample a few SHAP explanations,
     and report a HIT / PARTIAL / MISS verdict.
  5. Count single-packet flows with rst==1 or fin==1 (SEN-122 territory:
     training dropped these, the sensor doesn't; report only, don't filter).

Needs Linux or WSL with nfstream==6.6.0 (the sensor pins it); on bare
Windows this will refuse to start when sensor.package.load_package()
imports nfstream.

Writes:
  - results/sen-135/summary.csv   (one row per slice)
  - results/sen-135/<slice>.json  (full per-slice detail, incl. SHAP sample)
"""

from __future__ import annotations

# ruff: noqa: I001 — lightgbm must stay before pandas; see dev_model.py.

# See dev_model.py: lightgbm must be imported before pandas on this env.
import lightgbm  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sensor.package import load_package, score
from sensor.pcap_flows import flows_from_pcap
from sentinel_features.features import ATTACK_LABELS, BENIGN, LABELS
from sentinel_ml.explain import Explainer

RESULTS_DIR = Path("results") / "sen-135"
SLICE_CSV = Path("demo") / "slice.csv"
# Stage-3 gate / smoke / CTS demo slices come first.
PRIORITY_SLICES = [
    "PortScan-firewall-on",
    "PortScan-firewall-off",
    "SSH-Patator",
    "FTP-Patator",
]
SHAP_SAMPLE = 3  # top-3 reasons for a sample of alerts, per plan


def load_slice_catalog(path: Path = SLICE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["priority"] = df["slice_name"].apply(
        lambda n: PRIORITY_SLICES.index(n) if n in PRIORITY_SLICES else 999
    )
    return df.sort_values(["priority", "slice_name"]).reset_index(drop=True)


def _find_pcap(slice_dir: Path, slice_name: str) -> Path | None:
    """Look for <slice_name>.pcap in slice_dir; return None if missing."""
    candidate = slice_dir / f"{slice_name}.pcap"
    return candidate if candidate.exists() else None


def _attacker_mask(flows: pd.DataFrame, attacker: str, victim: str) -> np.ndarray:
    """True for flows between the slice's attacker_ip and victim_ip (either direction)."""
    if "src_ip" not in flows.columns or "dst_ip" not in flows.columns:
        raise KeyError("flows must have src_ip and dst_ip (pass keep_metadata=True)")
    s, d = flows["src_ip"].astype(str), flows["dst_ip"].astype(str)
    return ((s == attacker) & (d == victim)) | ((s == victim) & (d == attacker))


def _single_packet_rst_fin_mask(flows: pd.DataFrame) -> np.ndarray:
    """SEN-122 informational count: flows with 1 packet and either RST or FIN."""
    pkts = flows.get("bidirectional_packets")
    rst = flows.get("bidirectional_rst_packets")
    fin = flows.get("bidirectional_fin_packets")
    if pkts is None or rst is None or fin is None:
        return np.zeros(len(flows), dtype=bool)
    return (pkts == 1) & ((rst >= 1) | (fin >= 1))


def _verdict(attacker_alert_share: float, attacker_hit_class_share: float,
             expected_label: str) -> str:
    """HIT / PARTIAL / MISS using the plan's rough rules."""
    if attacker_alert_share < 0.20:
        return "MISS"
    if attacker_hit_class_share >= 0.60 and expected_label in LABELS:
        return "HIT"
    return "PARTIAL"


def check_slice(
    package: dict,
    slice_row: dict,
    pcap_path: Path,
    threshold: float,
    explainer: Explainer,
) -> dict:
    """Run one slice end-to-end and return a result dict."""
    flows = flows_from_pcap(pcap_path, keep_metadata=True)
    feature_cols = package["features"]
    detections = score(package, flows[feature_cols])
    detections = detections.reset_index(drop=True)
    flows = flows.reset_index(drop=True)

    is_attack_pred = detections["predicted_class"].isin(ATTACK_LABELS)
    is_alert = is_attack_pred & (detections["confidence"] >= threshold)

    attacker = slice_row["attacker_ip"]
    victim = slice_row["victim_ip"]
    expected_label = slice_row["label"]

    atk_mask = _attacker_mask(flows, attacker, victim)
    other_mask = ~atk_mask

    rstfin_mask = _single_packet_rst_fin_mask(flows)

    atk_detect = detections[atk_mask]
    class_mix = atk_detect["predicted_class"].value_counts().to_dict()
    hit_class_count = int((atk_detect["predicted_class"] == expected_label).sum())
    attacker_alerts = int(is_alert[atk_mask].sum())
    other_alerts = int(is_alert[other_mask].sum())

    attacker_alert_share = attacker_alerts / max(atk_mask.sum(), 1)
    attacker_hit_class_share = (
        hit_class_count / max(atk_mask.sum(), 1)
    )
    verdict = _verdict(attacker_alert_share, attacker_hit_class_share, expected_label)

    # SHAP sample for the first few alerts in the attacker set.
    sample_alert_idx = detections.index[(is_alert) & atk_mask][:SHAP_SAMPLE]
    shap_sample = []
    if len(sample_alert_idx):
        sample_flows = flows.loc[sample_alert_idx, feature_cols]
        sample_preds = detections.loc[sample_alert_idx, "predicted_class"].tolist()
        reasons = explainer.explain(sample_flows, sample_preds)
        for idx, r in zip(sample_alert_idx, reasons):
            shap_sample.append({
                "flow_index": int(idx),
                "class": r["class"],
                "top_3_labels": [x["label"] for x in r["reasons"][:3]],
            })

    return {
        "slice_name": slice_row["slice_name"],
        "expected_label": expected_label,
        "pcap": str(pcap_path),
        "flows_total": int(len(flows)),
        "flows_attacker_victim": int(atk_mask.sum()),
        "flows_other": int(other_mask.sum()),
        "attacker_class_mix": class_mix,
        "attacker_median_confidence": round(float(atk_detect["confidence"].median()), 4) if len(atk_detect) else None,
        "attacker_alerts": attacker_alerts,
        "attacker_alert_share": round(float(attacker_alert_share), 4),
        "attacker_hit_class_share": round(float(attacker_hit_class_share), 4),
        "other_alerts": other_alerts,
        "single_packet_rst_fin": int(rstfin_mask.sum()),
        "single_packet_rst_fin_predicted_classes": detections[rstfin_mask]["predicted_class"].value_counts().to_dict(),
        "shap_sample": shap_sample,
        "verdict": verdict,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--slices", type=Path, required=True,
                        help="Directory containing <slice_name>.pcap files")
    parser.add_argument("--threshold", type=float, required=True,
                        help="SEN-134 recommended threshold")
    parser.add_argument("--slice-csv", type=Path, default=SLICE_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_slice_catalog(args.slice_csv)
    package = load_package(args.package)
    explainer = Explainer(package)

    summary_rows = []
    missing = []
    for _, row in catalog.iterrows():
        pcap = _find_pcap(args.slices, row["slice_name"])
        if pcap is None:
            missing.append(row["slice_name"])
            continue
        print(f"[sen-135] running slice {row['slice_name']} ...", flush=True)
        try:
            result = check_slice(package, row.to_dict(), pcap, args.threshold, explainer)
        except Exception as e:
            print(f"  ERROR on {row['slice_name']}: {e!r}", file=sys.stderr)
            result = {"slice_name": row["slice_name"], "error": repr(e)}
        (args.out_dir / f"{row['slice_name']}.json").write_text(json.dumps(result, indent=2, default=str))
        summary_rows.append(result)

    summary_csv = args.out_dir / "summary.csv"
    if summary_rows:
        keys = [k for k in [
            "slice_name", "expected_label", "flows_total", "flows_attacker_victim",
            "flows_other", "attacker_alerts", "attacker_alert_share",
            "attacker_hit_class_share", "other_alerts",
            "attacker_median_confidence", "single_packet_rst_fin", "verdict",
        ]]
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"[sen-135] wrote {summary_csv}")

    if missing:
        print(f"[sen-135] MISSING PCAPs (expected in {args.slices}): {missing}", file=sys.stderr)
        # Flag the missing benign slice if we have none.
    benign_slice = catalog[catalog["label"] == BENIGN]
    if benign_slice.empty:
        print(
            "[sen-135] note: slice.csv has no benign slice. "
            "UC-1 ('benign replay -> no alerts') cannot be demonstrated yet.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

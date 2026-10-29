"""Plugins that label flows when regenerating the dataset.

PayloadManager and FlowLabelManager are taken from the authors' notebook
3-NFS-2023-TE-1-analyse-dataset.ipynb:
https://github.com/FlowFrontiers/CyberML-DataQuality/tree/main/4-NFS-2023/3-NFS-2023-TE-1-analyse-dataset.ipynb 
copied on 2026-10-28. Changes: import path only.

labeller.py in this folder is the authors' labeller, copied unchanged.

Paper: "Evaluating ML-Based Anomaly Detection Across Datasets of Varied
Integrity: A Case Study".
"""

from nfstream import NFPlugin

from sentinel_data.labeller import cicids2017


class PayloadManager(NFPlugin):
    """Manages the payload data for network flows."""

    def on_init(self, packet, flow):
        # Initialize payload sizes based on the packet direction.
        flow.udps.src2dst_payload = packet.payload_size if packet.direction == 0 else 0
        flow.udps.dst2src_payload = packet.payload_size if packet.direction == 1 else 0

    def on_update(self, packet, flow):
        # Update payload sizes based on the packet direction.
        flow.udps.src2dst_payload += packet.payload_size if packet.direction == 0 else 0
        flow.udps.dst2src_payload += packet.payload_size if packet.direction == 1 else 0

class FlowLabelManager(NFPlugin):
    """Labels flows upon expiration."""

    def __init__(self, day):
        self.day = day

    def on_expire(self, flow):
        # Assign a label to the flow and clean up payloads.
        flow.udps.label = cicids2017(
            self.day, flow, label_reverse=True, signal_reverse=False
        )
        self.cleanup_payload(flow)

    @staticmethod
    def cleanup_payload(flow):
        # Clean up payload data from the flow.
        if hasattr(flow.udps, "src2dst_payload"):
            del flow.udps.src2dst_payload
        if hasattr(flow.udps, "dst2src_payload"):
            del flow.udps.dst2src_payload
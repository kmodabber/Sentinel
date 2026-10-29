"""Plugins that exist only to label flows (dataset regeneration)."""
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
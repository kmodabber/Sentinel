"""NFStream settings and flow-expiry plugin.

The NFStreamer settings and FlowExpirationManager are adapted from the
authors' notebook 3-NFS-2023-TE-1-analyse-dataset.ipynb:
https://github.com/FlowFrontiers/CyberML-DataQuality/blob/main/4-NFS-2023/3-NFS-2023-TE-1-analyse-dataset.ipynb
Copied on 2026-10-28. Values unchanged. Wrapped in build_streamer().

Paper: "Evaluating ML-Based Anomaly Detection Across Datasets of Varied
Integrity: A Case Study".
"""

from nfstream import NFPlugin, NFStreamer

# The settings live in nfstream_settings.py so they can be read without
# nfstream installed; BPF is imported too so existing imports keep working.
from sentinel_features.nfstream_settings import BPF, SETTINGS  # noqa: F401


class FlowExpirationManager(NFPlugin):
    """Manages the expiration policy for TCP flows."""

    def on_init(self, packet, flow):
        # Set the expiration ID based on TCP rst or fin flags.
        if packet.rst or packet.fin:
            flow.expiration_id = -1

    def on_update(self, packet, flow):
        # Update expiration policy based on TCP rst or fin flags.
        if packet.rst or packet.fin:
            flow.expiration_id = -1

def build_streamer(source, extra_udps=()):
    """Build an NFStreamer with the dataset settings.

    extra_udps are appended after FlowExpirationManager, so the plugin
    order stays Expiration, Payload, Label as in the original notebook.
    """
    return NFStreamer(
        source=source,
        udps=[FlowExpirationManager(), *extra_udps],          # Default: None
        **SETTINGS,
    )
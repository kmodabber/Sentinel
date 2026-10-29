from nfstream import NFPlugin, NFStreamer

BPF = "ip and (ip proto \\tcp or \\udp)"  # IPv4 TCP and UDP only

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
        source=source
        , decode_tunnels=False                                # Default: True
        , bpf_filter=BPF                                      # Default: None
        , promiscuous_mode=True                               # Default: True
        , snapshot_length=1536                                # Default: 1536
        , idle_timeout=60                                     # Default: 120
        , active_timeout=120                                  # Default: 1800
        , accounting_mode=0                                   # Default: 0
        , udps=[FlowExpirationManager(), *extra_udps]         # Default: None
        , n_dissections=0                                     # Default: 20
        , statistical_analysis=True                           # Default: False
        , splt_analysis=0                                     # Default: 0
        , n_meters=1                                          # Default: 0
        , performance_report=0                                # Default: 0
    )
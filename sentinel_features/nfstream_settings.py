"""NFStream settings used to build NFS-2023-TE, as plain values.

Kept apart from nfstream_config.py so the model package and the sensor's
package loader can read them without importing nfstream itself. Values
are from the authors' notebook; see nfstream_config.py for the source.
"""

BPF = "ip and (ip proto \\tcp or \\udp)"  # IPv4 TCP and UDP only

SETTINGS = {
    "decode_tunnels": False,                                  # Default: True
    "bpf_filter": BPF,                                        # Default: None
    "promiscuous_mode": True,                                 # Default: True
    "snapshot_length": 1536,                                  # Default: 1536
    "idle_timeout": 60,                                       # Default: 120
    "active_timeout": 120,                                    # Default: 1800
    "accounting_mode": 0,                                     # Default: 0
    "n_dissections": 0,                                       # Default: 20
    "statistical_analysis": True,                             # Default: False
    "splt_analysis": 0,                                       # Default: 0
    "n_meters": 1,                                            # Default: 0
    "performance_report": 0,                                  # Default: 0
}

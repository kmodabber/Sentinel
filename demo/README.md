# Demo attack replay (TC-10)

One command replays a chosen attack slice into the sensor's virtual interface. Nothing leaves the machine.

```bash
./replay.sh list                  # slices in slice.csv, and whether each one has been cut
./replay.sh SSH-Patator           # replay one slice at 500 pps
PPS=2000 ./replay.sh DoS-Hulk     # faster
```

Runs on the Linux demo laptop (SEN-112). Requires `iproute2`, `ethtool`, `tcpreplay`, `tcpdump` and `wireshark-common` (`editcap`, `capinfos`), plus sudo.

## How it stays local

`replay.sh` sends only on `veth-replay`, which is paired with `veth-sensor`. That pair is a virtual cable with no route out. Before sending, the script aborts if any of these is true:

- either end is not a veth, or `veth-replay`'s peer isn't `veth-sensor`
- either end is attached to a bridge
- either end has an IP address
- the MTU is not 9000 (which would silently drop large packets)
  Slice names must exactly match a row in `slice.csv`, so a name can't be used to point at another path.

## Slices

`slice.csv` (SEN-110) is the source of truth. It has 11 slices: FTP-Patator, SSH-Patator, four DoS slices (slowloris, Slowhttptest, Hulk, GoldenEye), three web attacks (brute force, XSS, SQLi) and two port scans (firewall on and off).

The cutter uses these columns: `slice_name`, `date`, `pcap_file`, `start`, `end`, `attacker_ip` and `victim_ip`. Each slice keeps only the packets between the attacker and the victim (`host <attacker_ip> and host <victim_ip>`) inside the window. The slice PCAP is named `<slice_name>.pcap`.

### Time offset used to cut slices

The start and end times in `slice.csv` are the published CICIDS2017 label times. Those are lab local time (New Brunswick, ADT = **UTC−3** in July 2017). PCAP timestamps are UTC epoch, so:

> **PCAP time = label time + 3 hours** (`OFFSET_HOURS=3` in `cut_slices.sh`)

Example: SSH-Patator 14:00–15:12 local is cut as 17:00–18:12 UTC. Checked against the data: the slice's first attack packet is at 14:09 ADT (17:09 UTC) and its last at 15:11:26 ADT (18:11 UTC).

The published label windows can end before the attack does. SSH-Patator's published end is 15:00, but 27k attack packets come after it, so its `end` in `slice.csv` is 15:12. After cutting any slice, check that its last packet isn't right at the window edge.

### Where the cut slice PCAPs are stored

- **On the demo laptop:** `~/sentinel-slices/<name>.pcap`, outside the repo. Override with `SLICE_DIR=...`.
- **Master copies:** the team Drive. The full day PCAPs come from UNB.
- **Never in git.** `cut_slices.sh` refuses to write inside a git work tree, and `.gitignore` blocks `*.pcap` and `*.pcapng`.

### Re-cutting

```bash
PCAP_DIR=~/cicids2017-pcaps ./cut_slices.sh                       # cut any slice not cut yet
PCAP_DIR=~/cicids2017-pcaps FORCE=1 ./cut_slices.sh               # re-cut all
PCAP_DIR=~/cicids2017-pcaps ./cut_slices.sh PortScan-firewall-on  # (re)cut one
```

For each slice, the script cuts the time window with `editcap`, applies the BPF filter with `tcpdump`, and prints the packet count and the earliest/latest packet times. `capinfos` shows those times in the machine's local zone (EDT/EST in Toronto, not ADT). If the latest packet is within a second of the window end, the attack probably runs past it: extend `end` and re-cut with `FORCE=1`.

## TC-10 check

```bash
./tc10_check.sh SSH-Patator        # watches the real NIC (default route) during the replay
```

PASS requires both of these:

- tcpreplay reports `Failed packets: 0`
- **zero** packets from the CICIDS2017 lab addresses (172.16.0.1, 205.174.165.73, 192.168.10.0/24) appear on the laptop's real network card
  The result is saved to `/tmp/tc10-<slice>-*.txt`. Attach it to the ticket.

Last run (Nov 22, WSL2): SSH-Patator, 163,311 packets at 2000 pps, 0 failed, 0 seen on eth0, PASS. Still to repeat on the demo laptop (SEN-112).

## Packet size notes

- tcpreplay can't send frames larger than MTU + 14 bytes, so the veth pair runs at MTU 9000. This setting is re-applied on every run.
- Offloads (TSO/GSO/GRO/LRO) are turned off so the kernel doesn't split or merge replayed packets.
- `replay.sh` warns about oversized frames before sending and exits with code 2 if any packet fails.
- If you raise the MTU, also set NFStreamer's `snapshot_length` to at least the MTU (its default is 1536).

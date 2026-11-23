#!/usr/bin/env bash
# One-command, local-only replay of a demo attack slice (TC-10).
#
# Usage: ./replay.sh <slice-name>          e.g. ./replay.sh SSH-Patator
#        ./replay.sh list
#        PPS=2000 ./replay.sh <slice-name>          (override replay speed)
#        SLICE_DIR=/path ./replay.sh <slice-name>   (override slice location)
#
# Slices are cut by cut_slices.sh and live OUTSIDE the repo (default
# ~/sentinel-slices, master copies on the team Drive). Never commit PCAPs.
set -euo pipefail

TX=veth-replay      # tcpreplay writes here
RX=veth-sensor      # the sensor (NFStreamer) reads here
MTU=9000            # max frame tcpreplay can send = MTU + 14
PPS="${PPS:-500}"   # packets per second

HERE="$(cd "$(dirname "$0")" && pwd)"
CSV="$HERE/slice.csv"
SLICE_DIR="${SLICE_DIR:-$HOME/sentinel-slices}"

# slice names from the first CSV column (skip header, strip Windows line endings)
names() { tail -n +2 "$CSV" | tr -d '\r' | cut -d, -f1 | grep -v '^$'; }

list() {
  echo "Slices in slice.csv (looking in $SLICE_DIR):"
  names | while read -r n; do
    if [ -f "$SLICE_DIR/$n.pcap" ]; then echo "  $n"; else echo "  $n (PCAP not cut yet)"; fi
  done
}

[ -f "$CSV" ] || { echo "Missing $CSV"; exit 1; }
[ $# -eq 1 ] || { echo "usage: $0 <slice-name|list>"; list; exit 1; }
[ "$1" = "list" ] && { list; exit 0; }

[[ "$PPS" =~ ^[0-9]+$ ]] && [ "$PPS" -gt 0 ] || { echo "PPS must be a positive integer"; exit 1; }

# name must be a row in the CSV (exact match), which also blocks path tricks
names | grep -Fxq -- "$1" || { echo "Not in slice.csv: $1"; list; exit 1; }

PCAP="$SLICE_DIR/$1.pcap"
[ -f "$PCAP" ] || { echo "Slice '$1' is in slice.csv but its PCAP isn't at $PCAP (run cut_slices.sh)"; exit 1; }

for tool in ip ethtool tcpreplay; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool"; exit 1; }
done

# create the veth pair if it doesn't exist yet
if ! ip link show "$TX" >/dev/null 2>&1; then
  sudo ip link add "$TX" type veth peer name "$RX"
fi

# (re)apply settings on EVERY run, so a pair left over from an older run
# can't silently keep MTU 1500 or offloads that change packet sizes
for i in "$TX" "$RX"; do
  sudo ip link set "$i" mtu "$MTU" up
  sudo sysctl -qw "net.ipv6.conf.$i.disable_ipv6=1"
  # no checksum/segmentation offload on send, no merging (GRO/LRO) on receive
  sudo ethtool -K "$i" tx off tso off gso off gro off lro off >/dev/null 2>&1 || true
done
sudo ip link set "$RX" promisc on

# safety checks: refuse to send if either end could reach anything else
for i in "$TX" "$RX"; do
  ip -d link show "$i" | grep -q 'veth'    || { echo "ABORT: $i is not a veth"; exit 1; }
  ip -o link show "$i" | grep -q ' master ' && { echo "ABORT: $i is attached to a bridge"; exit 1; }
  [ -z "$(ip -o addr show dev "$i")" ]      || { echo "ABORT: $i has an IP address"; exit 1; }
  [ "$(cat "/sys/class/net/$i/mtu")" -ge "$MTU" ] || { echo "ABORT: $i MTU is not $MTU"; exit 1; }
done
# the peer of TX must be RX (not some other interface)
peer_idx="$(cat "/sys/class/net/$TX/iflink")"
[ "$peer_idx" = "$(cat "/sys/class/net/$RX/ifindex")" ] || { echo "ABORT: $TX is not paired with $RX"; exit 1; }

# size check: frames bigger than MTU+14 would be dropped by the kernel
if command -v tcpdump >/dev/null; then
  big="$(tcpdump -nn -r "$PCAP" "greater $((MTU + 15))" 2>/dev/null | wc -l)"
  [ "$big" -eq 0 ] || echo "WARNING: $big packets in $1 are larger than $((MTU + 14)) bytes and will fail"
fi

echo "Replaying '$1' -> $RX (local only, $PPS pps)"
out="$(sudo tcpreplay -i "$TX" --pps="$PPS" "$PCAP" 2>&1)" || { echo "$out"; exit 1; }
echo "$out"

failed="$(echo "$out" | sed -n 's/.*Failed packets:[[:space:]]*\([0-9]*\).*/\1/p' | head -n1)"
if [ -n "$failed" ] && [ "$failed" -ne 0 ]; then
  echo "ERROR: $failed packets failed to send (usually packet size vs MTU); replay is not faithful"
  exit 2
fi
echo "OK: '$1' replayed with 0 failed packets"

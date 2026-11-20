#!/usr/bin/env bash
# Usage: ./replay.sh <slice-name>     e.g. ./replay.sh SSH-Patator
#        ./replay.sh list
#        PPS=2000 ./replay.sh SSH-Patator     (override replay speed)
set -euo pipefail

TX=veth-replay      # tcpreplay writes here
RX=veth-sensor      # the sensor (NFStreamer) reads here
PPS="${PPS:-500}"   # packets per second

HERE="$(cd "$(dirname "$0")" && pwd)"
CSV="$HERE/slice.csv"
# SLICE_DIR="$HERE/../../slices"                      # real location (switch when decided)
SLICE_DIR="${SLICE_DIR:-$HERE/../models/slices}"             # testing with local pcap slice (change later)

# slice names from the first CSV column (skip header, strip Windows line endings)
names() { tail -n +2 "$CSV" | tr -d '\r' | cut -d, -f1; }

list() {
  echo "Slices in slice.csv:"
  names | while read -r n; do
    if [ -f "$SLICE_DIR/$n.pcap" ]; then echo "  $n"; else echo "  $n (PCAP not cut yet)"; fi
  done
}

[ -f "$CSV" ] || { echo "Missing $CSV"; exit 1; }
[ $# -eq 1 ] || { echo "usage: $0 <slice-name|list>"; list; exit 1; }
[ "$1" = "list" ] && { list; exit 0; }

# name must be a row in the CSV (exact match), which also blocks path tricks
names | grep -Fxq -- "$1" || { echo "Not in slice.csv: $1"; list; exit 1; }

PCAP="$SLICE_DIR/$1.pcap"
[ -f "$PCAP" ] || { echo "Slice '$1' is in slice.csv but its PCAP isn't at $PCAP"; exit 1; }

# create the veth pair if it doesn't exist yet
if ! ip link show "$TX" >/dev/null 2>&1; then
  sudo ip link add "$TX" type veth peer name "$RX"
  for i in "$TX" "$RX"; do
    sudo ip link set "$i" mtu 9000 up
    sudo sysctl -qw "net.ipv6.conf.$i.disable_ipv6=1"
    sudo ethtool -K "$i" tx off >/dev/null 2>&1 || true
  done
  sudo ip link set "$RX" promisc on
fi

# safety checks: refuse to send if either end could reach anything else
for i in "$TX" "$RX"; do
  ip -d link show "$i" | grep -q 'veth'    || { echo "ABORT: $i is not a veth"; exit 1; }
  ip -o link show "$i" | grep -q ' master ' && { echo "ABORT: $i is attached to a bridge"; exit 1; }
  [ -z "$(ip -o addr show dev "$i")" ]      || { echo "ABORT: $i has an IP address"; exit 1; }
done

echo "Replaying '$1' -> $RX (local only, $PPS pps)"
sudo tcpreplay -i "$TX" --pps="$PPS" "$PCAP"
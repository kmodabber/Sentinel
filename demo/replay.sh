#!/usr/bin/env bash
# Usage: ./replay.sh <slice-name>     e.g. ./replay.sh ssh_brute_force
#        ./replay.sh list
set -euo pipefail

TX=veth-replay      # tcpreplay writes here
RX=veth-sensor      # the sensor (NFStreamer) reads here
# SLICE_DIR="$(cd "$(dirname "$0")" && pwd)/../../slices"
SLICE_DIR="$(cd "$(dirname "$0")" && pwd)/../models" # Testing with local pcap slice (change later)

list() { echo "Available slices:"; ls "$SLICE_DIR" | sed 's/\.pcap$//' | sed 's/^/  /'; }

[ $# -eq 1 ] || { echo "usage: $0 <slice-name|list>"; list; exit 1; }
[ "$1" = "list" ] && { list; exit 0; }

# slice name: letters, digits, _ and - only (no paths)
[[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid slice name"; exit 1; }
PCAP="$SLICE_DIR/$1.pcap"
[ -f "$PCAP" ] || { echo "No such slice: $1"; list; exit 1; }

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

echo "Replaying '$1' -> $RX (local only)"
sudo tcpreplay -i "$TX" --pps=500 "$PCAP"
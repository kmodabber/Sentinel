#!/usr/bin/env bash
# TC-10 evidence: replay a slice while watching the real network card,
# and prove no replayed packet left the machine.
#
# Usage: ./tc10_check.sh <slice-name> [nic]   (nic defaults to the default-route interface)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SLICE="${1:?usage: $0 <slice-name> [nic]}"
NIC="${2:-$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')}"
[ -n "$NIC" ] || { echo "Could not find the real NIC; pass it as the 2nd argument"; exit 1; }

# CICIDS2017 lab addresses: these must never appear on the demo laptop's real network
WATCH='host 172.16.0.1 or host 205.174.165.73 or net 192.168.10.0/24'
LOG="$(mktemp /tmp/tc10-"$SLICE"-XXXX.txt)"
CAPDIR="$(mktemp -d /tmp/tc10-nic-XXXX)"
CAP="$CAPDIR/nic.pcap"
ERR="$CAP.err"

sudo -v   # ask for the sudo password up front, not in the middle of the run

# how long the replay will take, so tcpdump can stop on its own (no killing needed)
SLICE_DIR="${SLICE_DIR:-$HOME/sentinel-slices}"
PPS="${PPS:-500}"
PKTS=$(capinfos -c -M "$SLICE_DIR/$SLICE.pcap" 2>/dev/null | awk -F: '/Number of packets/ {gsub(/ /,"",$2); print $2}')
[ -n "$PKTS" ] || { echo "Can't read $SLICE_DIR/$SLICE.pcap (cut it first)"; exit 1; }
WATCH_SECS=$(( (PKTS + PPS - 1) / PPS + 15 ))
echo "Watching $NIC for ${WATCH_SECS}s while replaying $PKTS packets at $PPS pps"

tx_before=$(cat "/sys/class/net/$NIC/statistics/tx_packets")
# -Z root: stay root so tcpdump can write our temp file (by default it drops
# to the 'tcpdump' user, can't open the file, and exits immediately)
sudo timeout --signal=INT "$WATCH_SECS" tcpdump -ni "$NIC" -Z root -U -w "$CAP" "$WATCH" 2>"$ERR" &
TD=$!
sleep 2
kill -0 "$TD" 2>/dev/null || { echo "ABORT: NIC capture on $NIC didn't start:"; cat "$ERR"; exit 1; }

status=0
PPS="$PPS" "$HERE/replay.sh" "$SLICE" | tee "$LOG" || status=$?

echo "Replay done; waiting for the $NIC capture to finish..."
wait "$TD" 2>/dev/null || true
tx_after=$(cat "/sys/class/net/$NIC/statistics/tx_packets")
[ -s "$CAP" ] || { echo "FAIL: NIC capture file is empty/unreadable, leak check not done"; cat "$ERR"; exit 1; }
leaked=$(sudo tcpdump -nn -r "$CAP" 2>/dev/null | wc -l || true)
sent=$(sed -n 's/.*Actual: \([0-9]*\) packets.*/\1/p' "$LOG" | head -n1)

{
  echo "=== TC-10 result: $(date -Is) on $(hostname) ==="
  echo "slice:              $SLICE"
  echo "packets replayed:   ${sent:-?} (to veth-sensor)"
  echo "real NIC watched:   $NIC  (filter: $WATCH)"
  echo "slice packets seen on $NIC: $leaked"
  echo "$NIC tx_packets during run: $((tx_after - tx_before)) (normal background traffic, for context)"
} | tee -a "$LOG"

if [ "$status" -ne 0 ]; then echo "FAIL: replay.sh exited $status" | tee -a "$LOG"; exit "$status"; fi
if [ "$leaked" -ne 0 ]; then echo "FAIL: replayed traffic seen on $NIC" | tee -a "$LOG"; exit 1; fi
echo "PASS: nothing left the machine. Evidence saved to $LOG" | tee -a "$LOG"
sudo rm -rf "$CAPDIR"

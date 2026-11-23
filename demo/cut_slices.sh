#!/usr/bin/env bash
# Cut the demo slices listed in slice.csv out of the full CICIDS2017 PCAPs.
#
# Usage: ./cut_slices.sh              cut every slice that isn't cut yet
#        ./cut_slices.sh <name>       (re)cut one slice
#        FORCE=1 ./cut_slices.sh      re-cut everything
#
# Env:   PCAP_DIR   where the full day PCAPs are   (default ~/cicids2017-pcaps)
#        SLICE_DIR  where cut slices are written   (default ~/sentinel-slices)
#        OFFSET_HOURS  label-time -> PCAP-time offset (default 3)
#
# TIME OFFSET: the slice.csv start/end times are the published CICIDS2017
# label times, which are lab local time (New Brunswick, ADT = UTC-3 in July
# 2017). PCAP timestamps are UTC epoch. So:  PCAP time = label time + 3 h.
#
# slice.csv columns used (SEN-110; found by header name, order doesn't matter):
#   slice_name, date (YYYY-MM-DD), pcap_file, start (HH:MM), end (HH:MM),
#   attacker_ip, victim_ip   -> slice keeps only traffic between those two hosts
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CSV="$HERE/slice.csv"
PCAP_DIR="${PCAP_DIR:-$HOME/cicids2017-pcaps}"
SLICE_DIR="${SLICE_DIR:-$HOME/sentinel-slices}"
OFFSET_HOURS="${OFFSET_HOURS:-3}"

for tool in editcap tcpdump capinfos date; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool (apt install wireshark-common tcpdump)"; exit 1; }
done
[ -f "$CSV" ] || { echo "Missing $CSV"; exit 1; }

# refuse to write slices inside a git work tree
if git -C "$(dirname "$SLICE_DIR")" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ABORT: SLICE_DIR ($SLICE_DIR) is inside a git repo; slices must never be in git"; exit 1
fi
mkdir -p "$SLICE_DIR"

# column index by header name, so column order in slice.csv doesn't matter
header="$(head -n1 "$CSV" | tr -d '\r')"
col() {
  local i=1 f
  IFS=, read -ra fields <<< "$header"
  for f in "${fields[@]}"; do [ "$f" = "$1" ] && { echo "$i"; return; }; i=$((i + 1)); done
  echo "slice.csv has no '$1' column (header: $header)" >&2; exit 1
}
C_NAME=$(col slice_name); C_DATE=$(col date); C_SRC=$(col pcap_file)
C_START=$(col start); C_END=$(col end); C_ATK=$(col attacker_ip); C_VIC=$(col victim_ip)

# label (local) time -> UTC epoch seconds
to_epoch() { echo $(( $(date -u -d "$1" +%s) + OFFSET_HOURS * 3600 )); }

cut_one() {
  local name="$1" src="$2" start="$3" end="$4" bpf="$5"
  local out="$SLICE_DIR/$name.pcap" in="$PCAP_DIR/$src"
  [ -f "$in" ] || { echo "  SKIP $name: source PCAP not found: $in"; return 0; }
  if [ -f "$out" ] && [ "${FORCE:-0}" != 1 ] && [ -z "${ONLY:-}" ]; then
    echo "  have $name (FORCE=1 to re-cut)"; return 0
  fi

  local a b tmp
  a=$(to_epoch "$start"); b=$(to_epoch "$end")
  tmp="$(mktemp "$SLICE_DIR/.$name.XXXXXX.pcap")"
  echo "  cutting $name: $start -> $end local = $(date -u -d "@$a" '+%F %T') -> $(date -u -d "@$b" '+%T') UTC"

  # 1) time window, written as plain pcap (tcpreplay-friendly)
  editcap -F pcap -A "$a" -B "$b" "$in" "$tmp"
  # 2) optional BPF filter to keep only the attack conversation
  if [ -n "$bpf" ]; then
    tcpdump -nn -r "$tmp" -w "$out" "$bpf" 2>/dev/null
    rm -f "$tmp"
  else
    mv "$tmp" "$out"
  fi

  local n
  n=$(capinfos -c -M "$out" | awk -F: '/Number of packets/ {gsub(/ /,"",$2); print $2}')
  [ "${n:-0}" -gt 0 ] || { echo "  WARNING: $name is EMPTY - check times, offset and filter"; return 0; }
  capinfos -c -l -a -e -M "$out" | sed 's/^/    /' | grep -E 'packets|Earliest|Latest|limit'
}

echo "Cutting slices from $PCAP_DIR into $SLICE_DIR (offset +${OFFSET_HOURS} h)"
ONLY="${1:-}"
found=0
while IFS= read -r line; do
  line="${line%$'\r'}"; [ -z "$line" ] && continue
  IFS=, read -ra f <<< "$line"
  name="${f[$((C_NAME-1))]:-}"
  [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
  found=1
  d="${f[$((C_DATE-1))]:-}"; atk="${f[$((C_ATK-1))]:-}"; vic="${f[$((C_VIC-1))]:-}"
  bpf=""
  [ -n "$atk" ] && [ -n "$vic" ] && bpf="host $atk and host $vic"
  cut_one "$name" "${f[$((C_SRC-1))]:-}" "$d ${f[$((C_START-1))]:-}" "$d ${f[$((C_END-1))]:-}" "$bpf"
done < <(tail -n +2 "$CSV")

[ "$found" = 1 ] || { echo "No slice named '$ONLY' in slice.csv"; exit 1; }

#!/usr/bin/env bash
# SEN-87 / SEN-FR-30: weekly honeypot health check (plan section 9).
#
#   ./health_check.sh honeypot.conf
#
# Runs from a team machine over admin SSH and checks the five items in section 9:
#   1. VM is up            2. disk under DISK_MAX_PCT    3. logs still arriving
#   4. outbound still denied                             5. new data pulled
# plus the services Cowrie, OpenCanary and tcpdump being active, and the tcpdump
# ring staying under its cap. It prints a line to paste into Discord and appends
# one line to CHECK_LOG as the rota record.
# Exit codes: 0 honeypot OK, 1 problem found, 2 config error.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$HERE/lib/common.sh"
load_config "${1:-}"

PROBLEMS=()
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; PROBLEMS+=("$1"); }
info() { printf '  info  %s\n' "$1"; }

echo "Sentinel honeypot health check  $(utc_now)  host=$HP_HOST"

# ---- Collect everything from the VM in one ssh call ---------------------------
FACTS="$(
  {
    declare -p SERVICES SOURCES DATA_MOUNT OUTBOUND_TEST_CMD
    cat <<'REMOTE'
echo "now $(date +%s)"
echo "uptime $(uptime 2>/dev/null)"
for s in "${SERVICES[@]}"; do
  st="$(systemctl is-active "$s" 2>/dev/null)"
  echo "svc $s ${st:-unknown}"
done
pct="$(df -P "$DATA_MOUNT" 2>/dev/null | awk 'NR==2 {sub("%", "", $5); print $5}')"
echo "disk ${pct:-unknown}"
for src in "${SOURCES[@]}"; do
  IFS='|' read -r name dir glob stable <<<"$src"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name "$glob" -printf '%Ts %s\n' 2>/dev/null \
      | awk -v n="$name" '{c++; t+=$2; if ($1>m) m=$1} END {print "src", n, (c ? m : "none"), t+0, c+0}'
  else
    echo "src $name missing 0 0"
  fi
done
bash -c "$OUTBOUND_TEST_CMD" >/dev/null 2>&1
rc=$?
if (( rc == 0 )); then echo "outbound OPEN"
elif (( rc == 126 || rc == 127 )); then echo "outbound ERROR $rc"
else echo "outbound BLOCKED $rc"; fi
REMOTE
  } | remote_bash 2>/dev/null
)"
ssh_rc=$?

declare -A SVC=() SRC_NEWEST=() SRC_BYTES=() SRC_COUNT=()
vm_now="" disk="" outbound="" outbound_rc="" uptime_txt=""
while read -r key a b c d; do
  case "$key" in
    now) vm_now="$a" ;;
    uptime) uptime_txt="$a $b $c $d" ;;
    svc) SVC["$a"]="$b" ;;
    disk) disk="$a" ;;
    src) SRC_NEWEST["$a"]="$b"; SRC_BYTES["$a"]="$c"; SRC_COUNT["$a"]="$d" ;;
    outbound) outbound="$a"; outbound_rc="${b:-}" ;;
  esac
done <<<"$FACTS"

# ---- 1. VM up ------------------------------------------------------------------
if [[ -z "$vm_now" ]]; then
  fail "1 VM up: could not reach the VM over admin SSH (exit $ssh_rc). Check it is running in the Azure portal and your IP is on the admin list"
  for item in "2 disk" "3 logs arriving" "4 outbound denied"; do fail "$item: not checked, VM unreachable"; done
else
  pass "1 VM up: reached over admin SSH"
  info "uptime: ${uptime_txt# }"

  # ---- services ---------------------------------------------------------------
  for s in "${SERVICES[@]}"; do
    if [[ "${SVC[$s]:-unknown}" == active ]]; then pass "service $s active"
    else fail "service $s is ${SVC[$s]:-unknown}"; fi
  done

  # ---- 2. disk -------------------------------------------------------------------
  if [[ "$disk" =~ ^[0-9]+$ ]] && (( disk < DISK_MAX_PCT )); then
    pass "2 disk: ${disk}% used on $DATA_MOUNT (limit ${DISK_MAX_PCT}%)"
  else
    fail "2 disk: ${disk:-unknown}% used on $DATA_MOUNT (limit ${DISK_MAX_PCT}%)"
  fi
  if [[ -n "${PCAP_SOURCE:-}" && -n "${SRC_BYTES[$PCAP_SOURCE]:-}" ]]; then
    pcap_mb=$(( SRC_BYTES[$PCAP_SOURCE] / 1000000 ))   # tcpdump -C counts millions of bytes
    if (( SRC_BYTES[$PCAP_SOURCE] <= PCAP_CAP_MB * 1000000 )); then
      pass "2 pcap ring: ${pcap_mb} MB in ${SRC_COUNT[$PCAP_SOURCE]} files (cap ${PCAP_CAP_MB} MB)"
    else
      fail "2 pcap ring: ${pcap_mb} MB is over the ${PCAP_CAP_MB} MB cap, so tcpdump rotation may be broken"
    fi
  fi

  # ---- 3. logs still arriving ------------------------------------------------------
  for src in "${SOURCES[@]}"; do
    parse_source "$src"
    newest="${SRC_NEWEST[$SRC_NAME]:-missing}"
    if [[ "$newest" == missing ]]; then fail "3 $SRC_NAME: directory $SRC_DIR not found"
    elif [[ "$newest" == none ]]; then fail "3 $SRC_NAME: no files matching $SRC_GLOB in $SRC_DIR"
    else
      age_h=$(( (vm_now - newest) / 3600 ))
      if (( age_h < FRESH_HOURS )); then pass "3 $SRC_NAME: newest file ${age_h}h old"
      else fail "3 $SRC_NAME: newest file is ${age_h}h old (limit ${FRESH_HOURS}h)"; fi
    fi
  done

  # ---- 4. outbound denied ----------------------------------------------------------
  case "$outbound" in
    BLOCKED) pass "4 outbound denied: test connection from the VM failed as expected (exit $outbound_rc)" ;;
    OPEN)    fail "4 outbound denied: the VM REACHED THE INTERNET. Outbound deny rule is missing or broken. Use the kill switch now" ;;
    ERROR)   fail "4 outbound denied: test command could not run (exit $outbound_rc), so check OUTBOUND_TEST_CMD" ;;
    *)       fail "4 outbound denied: no result" ;;
  esac
  # Optional: confirm the NSG rule itself if the Azure CLI is set up.
  if [[ -n "${AZ_RESOURCE_GROUP:-}" && -n "${AZ_NSG_NAME:-}" ]] && command -v az >/dev/null; then
    rule="$(az network nsg rule show -g "$AZ_RESOURCE_GROUP" --nsg-name "$AZ_NSG_NAME" \
            -n "${AZ_DENY_RULE:-Deny-All-Outbound}" --query '[access,direction]' -o tsv 2>/dev/null | tr '\n' ' ')"
    if [[ "$rule" == "Deny Outbound "* ]]; then pass "4 NSG rule ${AZ_DENY_RULE:-Deny-All-Outbound} present (Deny, Outbound)"
    else fail "4 NSG rule ${AZ_DENY_RULE:-Deny-All-Outbound} not found or not Deny/Outbound: '${rule}'"; fi
  else
    info "4 also confirm in the Azure portal: NSG outbound rule Deny-All-Outbound, priority 100, Deny"
  fi
fi

# ---- 5. new data pulled ------------------------------------------------------------
STATE="$PULL_DEST/.pulled.tsv"
if [[ ! -f "$PULL_DEST/.last_pull" ]]; then
  fail "5 data pulled: no pull has ever run into $PULL_DEST"
else
  last="$(cat "$PULL_DEST/.last_pull")"; now="$(date +%s)"
  age_d=$(( (now - last) / 86400 ))
  window=$(( PULL_MAX_AGE_DAYS * 86400 ))
  recent=0
  [[ -f "$STATE" ]] && recent="$(awk -F'\t' -v cut=$((now - window)) '$6 >= cut' "$STATE" | wc -l)"
  if (( now - last > window )); then
    fail "5 data pulled: last pull was ${age_d} days ago (limit ${PULL_MAX_AGE_DAYS}). Run pull_data.sh and check its cron job"
  elif (( recent == 0 )); then
    fail "5 data pulled: pulls ran but copied no new files in ${PULL_MAX_AGE_DAYS} days"
  else
    pass "5 data pulled: last pull $(date -d "@$last" '+%F %H:%M'), $recent files copied in the last ${PULL_MAX_AGE_DAYS} days"
  fi
  if [[ -n "$vm_now" ]]; then
    pending="$("$HERE/pull_data.sh" --dry-run "$1" 2>/dev/null | awk '/^PENDING/ {print $2" files, "int($3/1000000)" MB"}')"
    [[ -n "$pending" ]] && info "5 waiting on the VM for the next pull: $pending"
  fi
fi

# ---- Result -------------------------------------------------------------------------
echo
if (( ${#PROBLEMS[@]} == 0 )); then
  result="OK"; discord="honeypot OK"
else
  result="PROBLEM"
  discord="honeypot PROBLEM: $(printf '%s; ' "${PROBLEMS[@]}")"; discord="${discord%; }"
fi
echo "Post in Discord:"
echo "  $discord"
if [[ "$result" == PROBLEM ]]; then
  echo
  echo "If something looks wrong, stop the VM first and ask questions later (plan section 10):"
  echo "  1. Azure portal: Stop (deallocate) the VM   2. add a deny-all-inbound NSG rule"
  echo "  3. post in Discord what happened and why     4. leave the VM and data untouched until Kam decides"
fi

mkdir -p "$(dirname "$CHECK_LOG")"
printf '%s\tchecker=%s\tresult=%s\tproblems=%d\t%s\n' \
  "$(utc_now)" "${USER:-unknown}" "$result" "${#PROBLEMS[@]}" "$discord" >>"$CHECK_LOG"

[[ "$result" == OK ]]

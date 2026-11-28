#!/usr/bin/env bash
# Test health_check.sh and pull_data.sh on a day of fake data, without a VM.
# HP_HOST=local makes the scripts run their "remote" side locally against a
# fake VM tree, and tests/bin/systemctl stands in for systemd.
#
#   bash tests/run_tests.sh            (uses a temp dir, removed afterwards)

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
VM="$T/vm"; STORE="$T/private"
export PATH="$HERE/tests/bin:$PATH"

passed=0; failed=0
ok()  { echo "ok    $1"; passed=$((passed + 1)); }
bad() { echo "FAIL  $1"; failed=$((failed + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; echo "      condition: $2"; fi; }

python3 -I "$HERE/tests/make_fake_day.py" "$VM" >/dev/null || { echo "fake data generation failed"; exit 1; }

# Base test config: the example with local mode and paths moved under $VM.
sed -e "s|^HP_HOST=.*|HP_HOST=\"local\"|" \
    -e "s|^HP_USER=.*|HP_USER=\"test\"|" \
    -e "s|^HP_PORT=.*|HP_PORT=\"0\"|" \
    -e "s|^REMOTE_SUDO=.*|REMOTE_SUDO=\"\"|" \
    -e "s|^PULL_DEST=.*|PULL_DEST=\"$STORE\"|" \
    -e "s#|/home/cowrie#|$VM/home/cowrie#" \
    -e "s#|/var/#|$VM/var/#" \
    -e "s|^DATA_MOUNT=.*|DATA_MOUNT=\"$VM\"|" \
    -e "s|^OUTBOUND_TEST_CMD=.*|OUTBOUND_TEST_CMD=\"timeout 3 bash -c 'exec 3<>/dev/tcp/192.0.2.1/443'\"|" \
    "$HERE/honeypot.conf.example" >"$T/base.conf"
conf() { cp "$T/base.conf" "$T/case.conf"; printf '%s\n' "$@" >>"$T/case.conf"; echo "$T/case.conf"; }
HC() { "$HERE/health_check.sh" "$@" >"$T/out" 2>&1; echo $?; }
PULL() { "$HERE/pull_data.sh" "$@" >"$T/out" 2>&1; echo $?; }
stored() { find "$STORE" -path "$STORE/.*" -prune -o -type f -path "*/$1" -print | wc -l; }

echo "== fresh honeypot, nothing pulled yet"
rc=$(HC "$T/base.conf")
check "health check fails before any pull"           '[[ $rc == 1 ]] && grep -q "FAIL  5 data pulled: no pull has ever run" "$T/out"'
check "items 1-4 and services pass"                  '! grep "FAIL" "$T/out" | grep -v "FAIL  5"'

echo "== dry run"
rc=$(PULL --dry-run "$T/base.conf")
expected=$(( $(find "$VM" -type f | wc -l) - 1 ))   # everything except the pcap tcpdump is writing
check "dry run lists every finished file ($expected)" '[[ $rc == 0 ]] && grep -q "^PENDING $expected " "$T/out"'
check "dry run copies nothing"                        '[[ ! -d $STORE/pcap ]]'
check "active pcap is skipped (stable 10 min)"        '! grep -q "capture.pcap06$" "$T/out"'

echo "== first pull"
rc=$(PULL "$T/base.conf")
check "pull succeeds and stores $expected files"      '[[ $rc == 0 ]] && grep -q "new=$expected replaced=0 dup=0" "$T/out"'
all_match=1
while IFS=$'\t' read -r p _ _ _ rel _; do cmp -s "$p" "$STORE/$rel" || { all_match=0; echo "      mismatch: $p"; }; done <"$STORE/.pulled.tsv"
check "every stored copy is byte-identical to the VM" '[[ $all_match == 1 ]]'
check "stored by source and day"                      '[[ -n $(find "$STORE/pcap" -mindepth 2 -name "*__capture.pcap00.gz") ]]'
check "private storage is mode 700"                   '[[ $(stat -c %a "$STORE") == 700 ]]'
check "pulled pcaps still decompress"                 'for f in $(find "$STORE/pcap" -name "*.gz"); do gzip -t "$f" || exit 1; done'

echo "== healthy honeypot after the pull"
rc=$(HC "$T/base.conf")
check "health check passes"                           '[[ $rc == 0 ]] && grep -q "^  honeypot OK$" "$T/out"'
check "result logged for the rota"                    'tail -1 "$STORE/health_checks.log" | grep -q "result=OK"'

echo "== second pull with nothing new"
rc=$(PULL "$T/base.conf")
check "nothing re-copied"                             '[[ $rc == 0 ]] && grep -q "new=0 replaced=0 dup=0" "$T/out"'

echo "== growing log"
echo '{"eventid":"cowrie.session.connect","src_ip":"203.0.113.77"}' >>"$VM/home/cowrie/cowrie/var/log/cowrie/cowrie.json"
rc=$(PULL "$T/base.conf")
check "grown cowrie.json replaces its shorter copy"   'grep -q "new=1 replaced=1" "$T/out" && [[ $(stored "*__cowrie.json") == 1 ]]'
check "replacement has the new line"                  'grep -rq 203.0.113.77 "$STORE/cowrie"'
check "state has one line for cowrie.json"            '[[ $(grep -c "/cowrie.json	" "$STORE/.pulled.tsv") == 1 ]]'

echo "== ring buffer wraps and overwrites capture.pcap00.gz"
python3 -I -c 'import gzip,sys; open(sys.argv[1],"wb").write(gzip.compress(b"\xd4\xc3\xb2\xa1"+bytes(20)+b"wrapped"))' \
  "$VM/var/log/honeypot/pcap/capture.pcap00.gz"
touch -d "-20 minutes" "$VM/var/log/honeypot/pcap/capture.pcap00.gz"
rc=$(PULL "$T/base.conf")
check "overwritten pcap is pulled as a new version"   'grep -q "new=1 replaced=0" "$T/out"'
check "both versions of capture.pcap00.gz kept"       '[[ $(stored "*__capture.pcap00.gz") == 2 ]]'

echo "== same content under a new name"
tty="$VM/home/cowrie/cowrie/var/lib/cowrie/tty"
first=$(ls "$tty" | head -1); cp -p "$tty/$first" "$tty/copy-of-$first"
rc=$(PULL "$T/base.conf")
check "duplicate content recorded, not stored twice"  'grep -q "new=0 replaced=0 dup=1" "$T/out"'

echo "== problems the check must catch"
conf >/dev/null
rc=$(FAKE_DOWN=cowrie HC "$T/case.conf")
check "service down"                                  '[[ $rc == 1 ]] && grep -q "FAIL  service cowrie is inactive" "$T/out"'
rc=$(HC "$(conf DISK_MAX_PCT=1)")
check "disk over the limit"                           '[[ $rc == 1 ]] && grep -q "FAIL  2 disk" "$T/out"'
rc=$(HC "$(conf PCAP_CAP_MB=0)")
check "pcap ring over its cap"                        '[[ $rc == 1 ]] && grep -q "FAIL  2 pcap ring" "$T/out"'
rc=$(HC "$(conf 'OUTBOUND_TEST_CMD="true"')")
check "outbound open"                                 '[[ $rc == 1 ]] && grep -q "REACHED THE INTERNET" "$T/out" && grep -q "Stop (deallocate)" "$T/out"'
rc=$(HC "$(conf 'OUTBOUND_TEST_CMD="no-such-tool x"')")
check "broken outbound test is not counted as blocked" '[[ $rc == 1 ]] && grep -q "test command could not run" "$T/out"'
cp -a "$VM" "$T/vm.bak"
find "$VM/var/tmp" -type f -exec touch -d "-30 hours" {} +
rc=$(HC "$T/base.conf")
check "opencanary log stopped arriving"               '[[ $rc == 1 ]] && grep -q "FAIL  3 opencanary: newest file is 30h old" "$T/out"'
rm -rf "$VM"; mv "$T/vm.bak" "$VM"
echo $(( $(date +%s) - 3 * 86400 )) >"$STORE/.last_pull"
rc=$(HC "$T/base.conf")
check "last pull too old"                             '[[ $rc == 1 ]] && grep -q "FAIL  5 data pulled: last pull was 3 days ago" "$T/out"'
rc=$(PULL "$T/base.conf")
rc=$(HC "$(conf "HP_HOST=\"nonexistent.invalid\"" 'HP_PORT="2"' "HP_KEY=\"$T/nokey\"")")
check "unreachable VM"                                '[[ $rc == 1 ]] && grep -q "FAIL  1 VM up" "$T/out"'
rc=$(HC "$(conf 'HP_HOST="TODO_VM_IP_OR_DNS"')")
check "unfilled config refused"                       '[[ $rc == 2 ]] && grep -q "TODO values" "$T/out"'
rc=$(PULL "$(conf "SOURCES+=(\"gone|$VM/no/such/dir|*|0\")")")
check "missing source dir warns, pull exits 1"        '[[ $rc == 1 ]] && grep -q "not found on VM" "$T/out"'

echo "== hostile VM: bad paths in the file listing"
H="$T/hostile"; mkdir -p "$H"
oc="$VM/var/tmp/opencanary.log"
printf '%s\t%s\t%s\t%s\n' \
  opencanary "$VM/var/tmp/../../../etc/passwd" 10 100 \
  opencanary "/etc/passwd" 10 100 \
  pcap "$VM/var/log/honeypot/pcap/sub/x.pcap" 10 100 \
  "../../evil" "$oc" 10 100 \
  opencanary "var/tmp/opencanary.log" 10 100 \
  opencanary "$oc" abc 100 \
  opencanary "$oc" "$(stat -c %s "$oc")" "$(stat -c %Y "$oc")" >"$H/listing"
rc=$(HOSTILE_LISTING="$H/listing" PATH="$HERE/tests/hostile/find_stub:$PATH" \
     PULL "$(conf "PULL_DEST=\"$H/store\"")")
check "6 bad listing lines rejected"                  '[[ $(grep -c "rejected listing line" "$T/out") == 6 ]]'
check "each rule fires" 'for r in "path contains .." "not directly inside" "unknown source name" "not an absolute path" "malformed line"; do grep -q "$r" "$T/out" || exit 1; done'
check "the one good line is still pulled, exit 1"     '[[ $rc == 1 ]] && grep -q "new=1 " "$T/out"'
check "nothing written outside PULL_DEST"             '[[ ! -e $T/evil && ! -e $H/evil && $(find "$H/store" -mindepth 1 -maxdepth 1 -type d ! -name ".*" -printf "%f\n") == opencanary ]]'

echo "== hostile VM: tar stream plants a symlink"
echo "do not touch" >"$H/outside_secret"
rc=$(HOSTILE_TARGET="$H/outside_secret" REAL_TAR="$(command -v tar)" \
     PATH="$HERE/tests/hostile/tar_stub:$PATH" PULL "$(conf "PULL_DEST=\"$H/store2\"")")
check "symlink rejected, pull exits 1"                '[[ $rc == 1 ]] && grep -q "extracted as a symlink" "$T/out"'
check "file it pointed at untouched and not copied"   '[[ $(cat "$H/outside_secret") == "do not touch" ]] && ! grep -rqs "do not touch" "$H/store2" --exclude-dir=".staging*"'
check "no symlinks or staging left in storage"        '[[ -z $(find "$H/store2" -type l) && -z $(find "$H/store2" -maxdepth 1 -name ".staging*") ]]'

echo "== overlapping SOURCES entries"
rc=$(PULL "$(conf "SOURCES+=(\"cowrie-again|$VM/home/cowrie/cowrie/var/log/cowrie|cowrie.json*|0\")" "PULL_DEST=\"$T/overlap\"")")
check "a file matched twice is pulled once, no warnings" '[[ $rc == 0 ]] && grep -q "warnings=0" "$T/out" && [[ $(find "$T/overlap" -name "*__cowrie.json" | wc -l) == 1 ]]'

echo "== bash version guard"
for s in health_check.sh pull_data.sh; do
  rc=$(sh "$HERE/$s" "$T/base.conf" >"$T/out" 2>&1; echo $?)
  check "$s refuses to run outside bash 4+"         '[[ $rc == 2 ]] && grep -q "needs bash 4 or newer" "$T/out"'
done

echo
echo "$passed passed, $failed failed"
(( failed == 0 ))

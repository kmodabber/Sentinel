#!/usr/bin/env bash
# Shared helpers for health_check.sh and pull_data.sh.

die() { echo "ERROR: $*" >&2; exit 2; }

load_config() {
  local conf="${1:-}"
  [[ -n "$conf" ]] || die "usage: $0 path/to/honeypot.conf"
  [[ -f "$conf" ]] || die "config not found: $conf"
  # shellcheck disable=SC1090
  source "$conf"
  local v
  for v in HP_HOST SOURCES PULL_DEST; do
    [[ -n "${!v:-}" ]] || die "$v is not set in $conf"
  done
  if grep -q 'TODO' <<<"$HP_HOST $HP_USER $HP_PORT $PULL_DEST"; then
    die "config still has TODO values; fill in HP_HOST, HP_USER, HP_PORT, PULL_DEST"
  fi
  REMOTE_SUDO="${REMOTE_SUDO-sudo -n}"
  PULL_MAX_AGE_DAYS="${PULL_MAX_AGE_DAYS:-2}"
  CHECK_LOG="${CHECK_LOG:-$PULL_DEST/health_checks.log}"
}

# remote_bash: run stdin as a bash script on the VM, with sudo if configured.
remote_bash() { run_remote "${REMOTE_SUDO:+$REMOTE_SUDO }bash -s"; }

# run_remote "command string" : runs on the VM (or locally when HP_HOST=local).
# stdin is passed through, so `... | run_remote 'bash -s'` works.
run_remote() {
  if [[ "$HP_HOST" == "local" ]]; then
    bash -c "$1"
  else
    ssh -p "$HP_PORT" -i "$HP_KEY" \
        -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes \
        "$HP_USER@$HP_HOST" "$1"
  fi
}

# Split "name|dir|glob|stable" into SRC_NAME SRC_DIR SRC_GLOB SRC_STABLE.
parse_source() {
  IFS='|' read -r SRC_NAME SRC_DIR SRC_GLOB SRC_STABLE <<<"$1"
  SRC_STABLE="${SRC_STABLE:-0}"
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

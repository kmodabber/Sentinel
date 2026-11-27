#!/usr/bin/env bash
# SEN-87 / SEN-FR-30: pull new honeypot data from the VM to private storage.
#
#   ./pull_data.sh honeypot.conf             pull everything new
#   ./pull_data.sh --dry-run honeypot.conf   list what would be pulled, copy nothing
#
# Runs on a team machine (Linux, macOS with GNU coreutils, or WSL). The VM can't
# connect out, so this pulls over admin SSH: one ssh call lists the files, a
# second streams only the new ones back as a tar. The VM needs only find and tar.
#
# Layout under PULL_DEST:
#   <source>/<YYYY-MM-DD>/<mtime UTC>__<file name>   pulled data (versioned)
#   .pulled.tsv   one line per pulled file: path, size, mtime, sha256, stored as, pulled at
#   .last_pull    epoch seconds of the last successful pull (read by health_check.sh)
#   pull.log      one summary line per run
#
# A file counts as new when its (path, size, mtime) has not been pulled before.
# If a new version only adds bytes to the end of the old one (a growing log), it
# replaces the old copy. Any other change (a pcap overwritten after the ring
# wraps) is kept alongside the old copy. Files with the same content are stored once.
# Exit codes: 0 ok, 1 pulled with problems, 2 config/connection error.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$HERE/lib/common.sh"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
load_config "${1:-}"

umask 077
mkdir -p "$PULL_DEST" || die "cannot create PULL_DEST $PULL_DEST"
chmod 700 "$PULL_DEST"
STATE="$PULL_DEST/.pulled.tsv"
touch "$STATE"

# ---- 1. List candidate files on the VM --------------------------------------
LISTING="$(
  {
    declare -p SOURCES
    cat <<'REMOTE'
for src in "${SOURCES[@]}"; do
  IFS='|' read -r name dir glob stable <<<"$src"
  stable="${stable:-0}"
  if [[ ! -d "$dir" ]]; then printf 'MISSING\t%s\t%s\n' "$name" "$dir"; continue; fi
  age=()
  (( stable > 0 )) && age=(-mmin "+$stable")
  find "$dir" -maxdepth 1 -type f -name "$glob" ${age[@]+"${age[@]}"} \
       -printf "$name\t%p\t%s\t%Ts\n"
done
REMOTE
  } | remote_bash
)" || die "could not list files on the VM (ssh failed or the remote script errored)"

# ---- 2. Load what has been pulled before --------------------------------------
declare -A SEEN=() LATEST_REL=() LATEST_SIZE=() SHA_REL=()
while IFS=$'\t' read -r p s m sha rel _at; do
  [[ -n "${p:-}" ]] || continue
  SEEN["$p|$s|$m"]=1
  LATEST_REL["$p"]="$rel"; LATEST_SIZE["$p"]="$s"
  SHA_REL["$sha"]="$rel"
done <"$STATE"

NEW_PATH=() NEW_SRC=()
new_bytes=0 warnings=0
while IFS=$'\t' read -r name path size mtime; do
  [[ -n "${name:-}" ]] || continue
  if [[ "$name" == MISSING ]]; then
    echo "WARN  source '$path' directory not found on VM: $size" >&2
    warnings=$((warnings + 1)); continue
  fi
  [[ -n "${SEEN["$path|$size|$mtime"]:-}" ]] && continue
  NEW_PATH+=("$path"); NEW_SRC+=("$name")
  new_bytes=$((new_bytes + size))
done <<<"$LISTING"

if (( DRY )); then
  for i in "${!NEW_PATH[@]}"; do printf 'NEW   %s  %s\n' "${NEW_SRC[$i]}" "${NEW_PATH[$i]}"; done
  printf 'PENDING %d %d\n' "${#NEW_PATH[@]}" "$new_bytes"
  exit 0
fi

LOCK="$PULL_DEST/.pull.lock"
mkdir "$LOCK" 2>/dev/null || die "another pull is running (delete $LOCK if it is stale)"
STAGING="$(mktemp -d "$PULL_DEST/.staging.XXXXXX")"
trap 'rm -rf "$LOCK" "$STAGING"' EXIT

now="$(date +%s)"
stored=0 dups=0 replaced=0 missing=0
REMOVED=()

if (( ${#NEW_PATH[@]} > 0 )); then
  # ---- 3. Stream the new files back in one tar ----------------------------------
  printf '%s\n' "${NEW_PATH[@]}" \
    | run_remote "${REMOTE_SUDO:+$REMOTE_SUDO }tar -cf - -P --no-recursion -T -" \
    | tar -xf - -C "$STAGING" 2> >(grep -v 'Removing leading' >&2)
  rc=("${PIPESTATUS[@]}")
  # remote tar: 1 = a file changed while being read (a growing log) -> fine.
  # >1 = e.g. a file rotated away between listing and copying -> that file is
  # skipped below and pulled next time; anything else still goes through.
  if (( rc[1] > 1 )); then
    echo "WARN  remote tar/ssh exit ${rc[1]}; files it could not read are retried next pull" >&2
    warnings=$((warnings + 1))
  fi
  (( rc[2] == 0 )) || die "local tar failed (exit ${rc[2]}); nothing was recorded"

  # ---- 4. File each pulled copy -------------------------------------------------
  for i in "${!NEW_PATH[@]}"; do
    path="${NEW_PATH[$i]}"; name="${NEW_SRC[$i]}"
    f="$STAGING$path"
    if [[ ! -f "$f" ]]; then missing=$((missing + 1)); continue; fi
    size="$(stat -c %s "$f")"; mtime="$(stat -c %Y "$f")"
    sha="$(sha256sum "$f" | cut -d' ' -f1)"

    if [[ -n "${SHA_REL[$sha]:-}" ]]; then        # same content already stored
      rel="${SHA_REL[$sha]}"; rm -f "$f"; dups=$((dups + 1))
    else
      day="$(date -u -d "@$mtime" +%F)"
      stamp="$(date -u -d "@$mtime" +%Y%m%dT%H%M%SZ)"
      rel="$name/$day/${stamp}__$(basename "$path")"
      n=1; base_rel="$rel"
      while [[ -e "$PULL_DEST/$rel" ]]; do rel="$base_rel.$n"; n=$((n + 1)); done

      old="${LATEST_REL[$path]:-}"
      if [[ -n "$old" && -f "$PULL_DEST/$old" ]] \
         && (( ${LATEST_SIZE[$path]} <= size )) \
         && cmp -s -n "${LATEST_SIZE[$path]}" "$PULL_DEST/$old" "$f"; then
        rm -f "$PULL_DEST/$old"; REMOVED+=("$old"); replaced=$((replaced + 1))
      fi
      mkdir -p "$PULL_DEST/$(dirname "$rel")"
      mv "$f" "$PULL_DEST/$rel"
      stored=$((stored + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$path" "$size" "$mtime" "$sha" "$rel" "$now" >>"$STATE"
    SEEN["$path|$size|$mtime"]=1
    LATEST_REL["$path"]="$rel"; LATEST_SIZE["$path"]="$size"; SHA_REL["$sha"]="$rel"
  done

  # Drop state lines for copies that were replaced by a longer version.
  if (( ${#REMOVED[@]} > 0 )); then
    printf '%s\n' "${REMOVED[@]}" >"$STAGING/removed"
    awk -F'\t' 'NR==FNR {gone[$0]=1; next} !($5 in gone)' "$STAGING/removed" "$STATE" >"$STAGING/state"
    mv "$STAGING/state" "$STATE"
  fi
fi

(( missing > 0 )) && { echo "WARN  $missing file(s) vanished before copying; retried next pull" >&2; warnings=$((warnings + 1)); }

echo "$now" >"$PULL_DEST/.last_pull"
summary="$(utc_now) pull new=$stored replaced=$replaced dup=$dups skipped=$missing warnings=$warnings bytes_listed=$new_bytes"
echo "$summary" >>"$PULL_DEST/pull.log"
echo "$summary"
(( warnings == 0 )) || exit 1

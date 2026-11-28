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

# declare -A needs bash 4+. Stock macOS ships bash 3.2 (see README "Requirements").
if [ -z "${BASH_VERSINFO:-}" ] || [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
  echo "ERROR: needs bash 4 or newer (this is ${BASH_VERSION:-not bash})." >&2
  echo "       On macOS: brew install bash coreutils, then run it with that bash." >&2
  exit 2
fi

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

# The listing comes from the VM, which faces the internet, so nothing in it is
# trusted: each line must name a configured source and a plain file directly
# inside that source's configured directory, or it is rejected.
declare -A ALLOWED_DIR=()
for src in "${SOURCES[@]}"; do
  parse_source "$src"
  ALLOWED_DIR["$SRC_NAME"]="${SRC_DIR%/}"
done
reject() { echo "WARN  rejected listing line from VM ($1): $2" >&2; warnings=$((warnings + 1)); }

NEW_PATH=() NEW_SRC=()
declare -A QUEUED=()
new_bytes=0 warnings=0
while IFS=$'\t' read -r name path size mtime extra; do
  [[ -n "${name:-}" ]] || continue
  if [[ "$name" == MISSING ]]; then
    echo "WARN  source '$path' directory not found on VM: $size" >&2
    warnings=$((warnings + 1)); continue
  fi
  line="$name $path"
  if [[ -z "${ALLOWED_DIR[$name]+set}" ]]; then reject "unknown source name" "$line"; continue; fi
  if [[ -n "${extra:-}" || ! "$size" =~ ^[0-9]+$ || ! "$mtime" =~ ^[0-9]+$ ]]; then
    reject "malformed line" "$line"; continue
  fi
  if [[ "$path" != /* ]]; then reject "not an absolute path" "$line"; continue; fi
  if [[ "/$path/" == *"/../"* || "/$path/" == *"/./"* || "$path" == *//* ]]; then
    reject "path contains .. or . or //" "$line"; continue
  fi
  base="${path##*/}"
  if [[ "${path%/*}" != "${ALLOWED_DIR[$name]}" || -z "$base" ]]; then
    reject "not directly inside ${ALLOWED_DIR[$name]}" "$line"; continue
  fi
  [[ -n "${SEEN["$path|$size|$mtime"]:-}" ]] && continue
  [[ -n "${QUEUED[$path]:-}" ]] && continue   # two SOURCES entries matched the same file
  QUEUED["$path"]=1
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
STAGING_REAL="$(realpath -- "$STAGING")"
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
    # Second guard, on the extracted file itself: it must be a regular file (not
    # a symlink a hostile tar stream could plant) that resolves inside staging.
    if [[ -L "$f" ]]; then
      echo "WARN  rejected $path: extracted as a symlink" >&2
      warnings=$((warnings + 1)); continue
    fi
    if [[ ! -f "$f" ]]; then missing=$((missing + 1)); continue; fi
    case "$(realpath -- "$f")" in
      "$STAGING_REAL"/*) ;;
      *) echo "WARN  rejected $path: resolves outside staging" >&2
         warnings=$((warnings + 1)); continue ;;
    esac
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

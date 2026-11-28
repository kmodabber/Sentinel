# SEN-87: honeypot health check and data pull

These scripts do the weekly health check (plan section 9) and the daily data pull (plan section 7) for the Sentinel honeypot (SEN-FR-27/30). Both run on a **team machine** over admin SSH. The VM can't connect out, so nothing is pushed from it.

| File | What it does |
|---|---|
| `health_check.sh` | The 5 weekly checks plus services and the pcap ring cap. Prints the line to post in Discord and logs the run. |
| `pull_data.sh` | Copies new Cowrie, OpenCanary and pcap files to private storage. Run it daily. `--dry-run` lists files without copying them. |
| `honeypot.conf.example` | Config template. Copy it to `honeypot.conf` and fill in the TODOs. |
| `tests/` | Fake-day generator and test suite (`bash tests/run_tests.sh`). No VM needed. |

Requirements: **bash 4 or newer**, plus `ssh`, `tar`, `sha256sum` and GNU `stat`, `date` and `realpath`. Linux and WSL have all of these. The scripts check the bash version first and stop with a message if it's older than 4.

- **macOS:** the built-in bash is 3.2, which lacks the associative arrays (`declare -A`) both scripts use, and the built-in `stat` and `date` are BSD versions. Run `brew install bash coreutils`, put Homebrew's `bin` and coreutils' `gnubin` ahead of `/bin` in `PATH` (the scripts use `#!/usr/bin/env bash`, so they pick up the first `bash` on `PATH`), and check with `bash --version`.

The VM needs only `find`, `tar`, `df`, `systemctl` and `curl`, which a standard Ubuntu image already has.

### What the scripts assume about the VM

Check these against the build. If one doesn't match, change the config, not the scripts.

| Assumption | Config setting | If your build differs |
|---|---|---|
| The admin user has **passwordless sudo** (the Azure default), so it can read files owned by the cowrie and tcpdump users | `REMOTE_SUDO="sudo -n"` | Give the admin user read access to every source and set `REMOTE_SUDO=""` |
| Cowrie, OpenCanary and the tcpdump ring each run as a **systemd service** | `SERVICES=(cowrie opencanary honeypot-tcpdump)` | Put your real unit names here. If Cowrie is started with `bin/cowrie start` and not systemd, the service check will always fail, so make it a systemd unit (recommended, so it restarts after a reboot) |
| Cowrie is installed in `/home/cowrie/cowrie` | `cowrie`, `cowrie-text`, `cowrie-tty` lines in `SOURCES` | Change the directory in those three lines |
| OpenCanary logs to `/var/tmp/opencanary.log` (its default) | `opencanary` line in `SOURCES` | Use the path in `/etc/opencanaryd/opencanary.conf` |
| tcpdump writes `capture.pcapNN` (gzipped to `.gz`) in `/var/log/honeypot/pcap`, for example `tcpdump -i eth0 -C 100 -W 50 -w /var/log/honeypot/pcap/capture.pcap 'not port <admin port>'` | `pcap` line in `SOURCES` | Change the directory or glob to match your tcpdump command |
| The data sits on the root disk | `DATA_MOUNT="/"` | Set it to the data disk's mount point if you add one |

## First run on the VM (do this once before go-live)

This is the "tested once" step for SEN-87. Run it from your own machine after Cowrie, OpenCanary and tcpdump are running (the dry run in plan section 11 is a good time: it also proves step 7 there, a pull working with outbound blocked).

1. **Fill in `honeypot.conf`** (see setup below), then check you can log in at all:
   `ssh -p <admin port> -i <key> <admin user>@<VM>`
2. **Dry-run the pull.** It only lists files and copies nothing:
   `./pull_data.sh --dry-run honeypot.conf`
   Expect `NEW ...` lines for each source and a final `PENDING <files> <bytes>`. A `WARN ... directory not found` line means that source's path is wrong; fix it in `SOURCES`.
3. **Pull for real:** `./pull_data.sh honeypot.conf`. Expect `pull new=<n> ... warnings=0`. Open a couple of the copied files under `PULL_DEST` to check they look right.
4. **Pull again.** Expect `new=0` (or only files that changed since step 3).
5. **Run the health check:** `./health_check.sh honeypot.conf`. Every line should be `PASS`, ending in `honeypot OK`. Before outbound is denied, item 4 will (correctly) FAIL.
6. **Send Hossane** the output of steps 2, 3 and 5, plus any config lines you changed, so the defaults can be fixed for everyone.

## Troubleshooting

| You see | Cause and fix |
|---|---|
| `config still has TODO values` | Fill in `HP_HOST`, `HP_USER`, `HP_PORT`, `PULL_DEST` in `honeypot.conf` |
| `FAIL 1 VM up` / `could not list files on the VM` | SSH failed. Check the VM is running, your current IP is on the admin NSG rule (`Allow-Admin-SSH`), and the port and key are right. Try the plain `ssh` command from step 1 |
| `sudo: a password is required` | The admin user lacks passwordless sudo; see the first row of the assumptions table |
| `WARN source '<name>' directory not found on VM` | That `SOURCES` path doesn't exist on the VM; fix the path |
| `FAIL service <x> is unknown` | Wrong unit name in `SERVICES` (`systemctl list-units --type=service` on the VM shows the real ones), or the service isn't run by systemd |
| `FAIL 4 outbound denied: test command could not run` | `curl` is missing on the VM. Install it before the outbound deny goes on, or change `OUTBOUND_TEST_CMD` |
| `another pull is running` | A pull is already running, or one crashed. If none is running, delete `PULL_DEST/.pull.lock` |

## One-time setup (Tareq, at build)

1. `cp honeypot.conf.example honeypot.conf` and fill in the values the plan left open (section 13):
   - `HP_HOST`, `HP_USER`, `HP_PORT`: the admin SSH port (not 22, 2222 or 2223), and `HP_KEY`: your key.
   - `PULL_DEST`: the private storage location. Only team members may be able to read it.
   - `SERVICES`: the real systemd unit names for Cowrie, OpenCanary and the tcpdump ring.
   - `SOURCES`: check the paths against the build. The OpenCanary log path is whatever its config sets, and the pcap dir and prefix are whatever the tcpdump unit uses.
2. **Never commit `honeypot.conf`.** It names the host and key. Add it to `.gitignore`.
3. Do a dry run from your machine: `./pull_data.sh --dry-run honeypot.conf`. It should list files and print `PENDING n bytes`.
4. Run one real pull and one health check. Both should end cleanly (`honeypot OK` once data exists).
5. Schedule the daily pull on a machine that is **on every day over the break**, since a pull that doesn't run lets old captures be overwritten. If that's a laptop, the rota person each week should run it (or check its cron job) while their laptop is on. For example with cron:
   ```
   15 6 * * *  /path/to/sen-87/pull_data.sh /path/to/honeypot.conf >>/path/to/pull-cron.log 2>&1
   ```
   The ring holds about 5 GB (`-C 100 -W 50`). If data arrives faster than one ring per day, pull more often.

## Weekly check (rota: Tareq and Arshan, alternating, Dec 1 to Jan 8)

About 15 minutes:

1. Run `./health_check.sh honeypot.conf`.
2. Post the line it prints under "Post in Discord" (`honeypot OK` or `honeypot PROBLEM: ...`).
3. In the Azure portal, confirm the NSG outbound rule **Deny-All-Outbound** (priority 100, Deny) is still there. The script proves outbound traffic fails from the VM, but only the portal shows the rule itself. If you set `AZ_RESOURCE_GROUP` and `AZ_NSG_NAME` and have `az` logged in, the script checks the rule too.
4. Glance at the cost against the budget alert.

Each run appends a line to `CHECK_LOG` (default `PULL_DEST/health_checks.log`). That file is the rota record.

### What each check means

| Check | Passes when | If it fails |
|---|---|---|
| 1 VM up | admin SSH connects | Check the VM is running in the portal and your IP is on the admin list |
| services | each `SERVICES` unit is `active` | `systemctl status <unit>` on the VM |
| 2 disk | data disk under `DISK_MAX_PCT` (70%) | Look for a log growing without rotation |
| 2 pcap ring | pcap files total no more than `PCAP_CAP_MB` | tcpdump rotation is broken, and the disk will fill |
| 3 logs arriving | every source has a file newer than `FRESH_HOURS` (24 h) | The service may be stuck. Check its unit and log |
| 4 outbound denied | `OUTBOUND_TEST_CMD` **fails** on the VM | **The VM reached the internet. Use the kill switch now** |
| 5 data pulled | last pull within `PULL_MAX_AGE_DAYS` (2) and new files copied | Run `pull_data.sh` by hand and fix its cron job |

**If anything looks wrong, stop the VM first and ask questions later** (plan section 10). Stop (deallocate) the VM in the portal, add a deny-all-inbound rule, post in Discord what happened, and leave everything untouched until Kam decides.

## How the pull stores data

```
PULL_DEST/
  pcap/2026-11-26/20261126T031500Z__capture.pcap03.gz
  cowrie/2026-11-25/20261125T235959Z__cowrie.json.2026-11-25
  cowrie-text/  cowrie-tty/  opencanary/
  .pulled.tsv   .last_pull   pull.log   health_checks.log
```

- Each file is filed under the source name and the day of its last change, with its mtime (UTC) as a prefix.
- **The ring buffer wrapping is safe.** When tcpdump overwrites `capture.pcap03.gz`, the new version is stored next to the old one, never over it.
- **Growing logs don't pile up.** If the new `cowrie.json` only adds lines to the copy already pulled, it replaces that copy.
- Files with identical content are stored once. The pcap tcpdump is still writing, and files changed in the last 10 minutes, wait for the next pull.
- Storage is created with mode 700, and files with umask 077.
- **Not pulled: Cowrie `downloads/`.** That directory holds files attackers upload, which may be malware, so it stays on the VM unless Kam approves copying it to a team machine.
- **Nothing the VM sends is trusted.** The VM faces the internet, so the pull treats its file listing and tar stream as hostile. A listed file is copied only if it names a configured source and sits directly inside that source's configured directory. Paths containing `..`, `.` or `//`, relative paths, subfolders, unknown source names and malformed lines are rejected with a `WARN`. After extraction, any file that arrived as a symlink, or that resolves outside the staging folder, is rejected before anything is moved or deleted.
- This is raw, private data. Anonymisation (SEN-FR-31, section 8) runs on it afterwards. None of it goes in git.

## Testing

`bash tests/run_tests.sh` builds a fake VM tree from one day of fake data (`tests/make_fake_day.py`), runs both scripts against it in local mode, and checks 39 cases:

- a clean pull and a re-pull, with byte-for-byte copies
- the active pcap skipped
- a growing log replacing its copy, a wrapped pcap kept as a second version, and duplicate content stored once
- a hostile VM: a listing with `..`, outside, subfolder, relative and unknown-source paths, and a tar stream that plants a symlink, all rejected with nothing written outside storage
- overlapping `SOURCES` entries pulling a file once, and both scripts refusing to run outside bash 4+
- each failure the health check should catch: service down, disk full, ring over cap, outbound open, broken outbound test, stale log, old pull, unreachable VM, unfilled config, and a missing source dir

Not covered by the tests (first real run on the VM):

- real SSH
- `sudo -n` reading the service users' files
- running under real macOS bash 3.2 (the version guard is tested by running the scripts under `sh`)
- real systemd unit names
- the `az` NSG check

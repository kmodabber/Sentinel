"""SEN-FR-01: download the NFS-2023-TE dataset and check it against data/SHA256SUMS.

Run from the repo root:
    python -m sentinel_data.get_data

The five zips (Monday to Friday) are saved and unzipped into data/raw/NFS-2023-TE/.
If a zip is already there it is not downloaded again, but it is still checked.
If any file doesn't match its checksum the script stops and names the file.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Download from a fixed commit of the dataset repo, not "main", so we always
# get the same files even if the authors update theirs later
DATASET_COMMIT = "c6e0191b2c5c058abb6c9de68aad555d59944879"
BASE_URL = (
    "https://raw.githubusercontent.com/FlowFrontiers/CyberML-DataQuality/"
    f"{DATASET_COMMIT}/4-NFS-2023/NFS-2023-TE/datasets"
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "NFS-2023-TE"
SUMS_FILE = DATA_DIR / "SHA256SUMS"


class ChecksumError(Exception):
    """Raised when a file is missing or its SHA-256 doesn't match."""


def read_sums(sums_file=SUMS_FILE):
    """Read SHA256SUMS into a dict of {filename: expected hash}.

    Same format as the sha256sum / shasum tools: "<hash>  <filename>".
    Blank lines and lines starting with # are skipped.
    """
    sums = {}
    for line in Path(sums_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest.lower()
    return sums


def sha256(path):
    """SHA-256 of a file, read in 1 MB chunks so big files don't fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    """Download url to dest."""
    # Save as .part first and rename at the end, so if the download is cut
    # off we never leave a half file under the real name
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(dest)


def verify(sums, raw_dir=RAW_DIR):
    """Check every file in sums. Raises ChecksumError listing all bad files."""
    bad = []
    for name, expected in sums.items():
        path = Path(raw_dir) / name
        if not path.exists():
            bad.append(f"{name}: missing")
            continue
        actual = sha256(path)
        if actual != expected:
            bad.append(f"{name}: expected {expected}, got {actual}")
        else:
            print(f"OK  {name}")
    if bad:
        raise ChecksumError("Checksum check failed:\n  " + "\n  ".join(bad))


def extract(sums, raw_dir=RAW_DIR):
    """Unzip each Day.csv.zip to Day.csv, skipping ones already unzipped."""
    for name in sums:
        if not name.endswith(".zip"):
            continue
        csv_name = name[: -len(".zip")]
        if (Path(raw_dir) / csv_name).exists():
            continue
        print(f"Extracting {name}")
        with zipfile.ZipFile(Path(raw_dir) / name) as zf:
            zf.extract(csv_name, raw_dir)


def get_data(base_url=BASE_URL, raw_dir=RAW_DIR, sums_file=SUMS_FILE):
    """Download any missing files, check all of them, then unzip."""
    sums = read_sums(sums_file)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in sums:
        if not (raw_dir / name).exists():
            download(f"{base_url}/{name}", raw_dir / name)

    # Unzip only after every file has passed the check
    verify(sums, raw_dir)
    extract(sums, raw_dir)
    print(f"All {len(sums)} files verified in {raw_dir}")


if __name__ == "__main__":
    try:
        get_data()
    except ChecksumError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Delete the bad file(s) and run again.", file=sys.stderr)
        sys.exit(1)

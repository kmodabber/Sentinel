"""TC-01: data download is verified (SEN-FR-01)."""
import hashlib
import zipfile

import pytest

from sentinel_data.get_data import ChecksumError, get_data


def make_source(tmp_path):
    """Make a fake download folder with one zipped CSV and a matching SHA256SUMS.

    get_data() is pointed at this folder with a file:// URL, so the tests
    never touch the internet.
    """
    src = tmp_path / "remote"
    src.mkdir()
    zpath = src / "Monday.csv.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("Monday.csv", "id,label\n0,BENIGN\n")
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"# test\n{digest}  Monday.csv.zip\n")
    return src, sums


def test_download_verifies_and_extracts(tmp_path):
    """A normal run downloads, passes the check and unzips the CSV."""
    src, sums = make_source(tmp_path)
    raw = tmp_path / "raw"
    get_data(base_url=src.as_uri(), raw_dir=raw, sums_file=sums)
    assert (raw / "Monday.csv.zip").exists()
    assert (raw / "Monday.csv").read_text().startswith("id,label")


def test_changed_byte_stops_and_names_file(tmp_path):
    """TC-01: change one byte in a downloaded file, the next run must fail and name it."""
    src, sums = make_source(tmp_path)
    raw = tmp_path / "raw"
    get_data(base_url=src.as_uri(), raw_dir=raw, sums_file=sums)

    # flip the last byte of the zip
    zpath = raw / "Monday.csv.zip"
    data = bytearray(zpath.read_bytes())
    data[-1] ^= 0xFF
    zpath.write_bytes(bytes(data))

    with pytest.raises(ChecksumError, match="Monday.csv.zip"):
        get_data(base_url=src.as_uri(), raw_dir=raw, sums_file=sums)


def test_missing_file_is_reported(tmp_path):
    """If a file can't be downloaded the script errors instead of carrying on."""
    src, sums = make_source(tmp_path)
    (src / "Monday.csv.zip").unlink()
    with pytest.raises(Exception):
        get_data(base_url=src.as_uri(), raw_dir=tmp_path / "raw", sums_file=sums)

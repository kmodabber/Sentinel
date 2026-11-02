# data

Nothing in here is committed except this README and `SHA256SUMS`.

## Getting the dataset (SEN-FR-01)

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m sentinel_data.get_data
```

This downloads NFS-2023-TE (corrected CICIDS2017, features built with NFStream)
from FlowFrontiers/CyberML-DataQuality, pinned to commit `c6e0191`, into
`data/raw/NFS-2023-TE/`. Every zip is checked against `SHA256SUMS`. If any file
doesn't match, the script stops, names the file and exits with code 1; delete
that file and run again. About 190 MB to download, 1.25 GB once unzipped.

## Layout

```
data/
  SHA256SUMS            checksums of the five zips (committed)
  raw/NFS-2023-TE/      Monday..Friday .csv.zip and .csv
  processed/            output of cleaning and the split
```

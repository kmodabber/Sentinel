# sentinel_data

Download, clean and split NFS-2023-TE. Run from the repo root, in order:

```
python -m sentinel_data.get_data   # SEN-FR-01  ~190 MB download, checksummed
python -m sentinel_data.clean      # SEN-FR-03  ~1.5 min, writes data/processed/clean.parquet
python -m sentinel_data.split      # SEN-FR-04  ~20 s, writes train/val/test.parquet + data_card.json
```

Tests: `pytest` (TC-01 `test_get_data.py`, TC-02 `test_clean.py`, TC-03 `test_split.py`).

## Cleaning rules (SEN-FR-03)

Applied in this order to all five days together:

1. `inf` / `-inf` become NaN, then any row with a NaN (or no label) is dropped.
2. Rows with a negative duration are dropped.
3. Exact duplicate rows are dropped, keeping the first. `id` is ignored here
   since it's just the row number in each day's file.
4. Identifier columns are dropped: `id`, `expiration_id`, IPs, MACs, OUIs,
   `src_port`, and every `*_first_seen_ms` / `*_last_seen_ms` except
   `bidirectional_first_seen_ms`.
5. Columns with a single value across the whole dataset are dropped.

`day` and `bidirectional_first_seen_ms` are kept so the split can order flows
by time, but they are **not features**. Use `feature_columns(df)` from
`sentinel_data.clean` to get the feature list.

Result on the full data (Nov 2):

| Step | Rows |
|---|---|
| Raw, 5 days | 3,153,360 |
| NaN / inf | 0 dropped |
| Negative duration | 0 dropped |
| Exact duplicates | 2,923 dropped |
| Clean | 3,150,437 |

Constant columns dropped: `ip_version`, `vlan_id`, `bidirectional_urg_packets`,
`src2dst_urg_packets`, `dst2src_urg_packets`. 56 features remain.

Note for modelling: 1.74M rows (55%) share their features and label with
another row (different IPs/times). They're real repeated flows (scans, floods)
so they're kept, but identical rows can sit in both train and test.

## Split (SEN-FR-04)

Each (day, label) is one window: one attack on one day, or one day's benign
traffic. Inside each window, flows are sorted by start time and cut at the
60% and 80% points (t1, t2):

- train: flow ends before t1 - gap
- val: flow starts at or after t1 and ends before t2 - gap
- test: flow starts at or after t2

Gap is 1 s by default (`--gap-seconds`). A flow that would be running across a
boundary is dropped, so no connection lands in two parts. Same input always
gives the same output.

Why not a fixed gap on start times: dense attacks (PortScan, DoS GoldenEye)
fit their middle 20% of flows into under two minutes, so a 120 s gap left
them with no validation data at all.

Result: train 1,887,703 / val 626,336 / test 630,181, 6,217 dropped at
boundaries. Per-class counts are in `data/processed/data_card.json`.

The output files are named the way `sentinel_ml/baselines.py` reads them
(`data/processed/train.parquet`, `val.parquet`, `test.parquet`).

## Rare classes (SEN-FR-05)

Attack classes with fewer than 100 flows after cleaning are merged into one
**"Rare attack"** class (`--rare-min-flows` to change it). Each one is still
split inside its own time window first, then relabelled, so every rare attack
appears in train, val and test. The original label is kept in
`label_original` (not a feature) in case the evaluation wants to look at them
one by one. Nothing is excluded.

| Merged class | train | val | test |
|---|---|---|---|
| Heartbleed | 5 | 1 | 3 |
| Infiltration | 16 | 6 | 6 |
| Web Attack - Sql Injection | 7 | 2 | 3 |
| Web Attack - XSS | 15 | 4 | 6 |
| **Rare attack (total)** | **43** | **13** | **18** |

Web Attack - Brute Force (147 flows) and Bot (736) stay as their own classes.
The merged list and counts are written to `rare_classes` in
`data/processed/data_card.json`; the evaluation report should quote it.

## Sharing the processed data (SEN-FR-06)

`split` saves `train.parquet`, `val.parquet`, `test.parquet` and
`data_card.json` in `data/processed/`. The data card has the version, date,
rows per class, the rare class merge, the feature list, the code commit that
made the files, and each file's SHA-256 and row count.

These files are too big for git, so they go in the shared Drive folder
**Sentinel > Data**. The version they belong to is in `data_card.json`.

If you download them from Drive instead of running the scripts, put all four
in `data/processed/` and check them:

```
python -m sentinel_data.check_processed
```

It prints `OK` with the row counts, or names any file that doesn't match the
data card.

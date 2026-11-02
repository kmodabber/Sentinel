# sentinel_data

Download and clean NFS-2023-TE. Run from the repo root, in order:

```
python -m sentinel_data.get_data   # SEN-FR-01  ~190 MB download, checksummed
python -m sentinel_data.clean      # SEN-FR-03  ~1.5 min, writes data/processed/clean.parquet
```

Tests: `pytest` (TC-01 `test_get_data.py`, TC-02 `test_clean.py`).

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

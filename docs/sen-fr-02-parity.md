# SEN-FR-02 / TC-04: Tuesday regeneration check

## Goal

Regenerate the CICIDS2017 Tuesday flows with our NFStream setup and check that
flow counts per label match the published dataset within 1% (SEN-FR-02).
Columns are also checked (first part of TC-04). Value-level matching and the
sensor run are left for Stage 2 (SEN-FR-16, 19).

## Method

- Input: Tuesday-WorkingHours.pcap, raw; From: https://cicresearch.ca/CICDataset/CIC-IDS-2017/
- NFStream 6.6.0, Python 3.12.10, native Windows
- Settings (unchanged from the authors): BPF `ip and (ip proto tcp or udp)`,
  idle timeout 60, active timeout 120, n_dissections 0, statistical analysis
  on, TCP flows expire on RST/FIN, single-packet RST/FIN flows dropped.
- Labeller: the authors' `labeller.py`, unmodified, `label_reverse=True`,
  `signal_reverse=False`.
- Reference: `Tuesday.csv` renamed to `CyberML-DataQuality-Tuesday.csv` ([NFS-2023-TE repo link](https://github.com/FlowFrontiers/CyberML-DataQuality/tree/main/4-NFS-2023/NFS-2023-TE/datasets)),
  stored at [Drive link](https://drive.google.com/drive/folders/1vlJpfTmcXEFZKp55Dmpa__f7Z3ZlDsTV?usp=drive_link).
- Code: `sentinel_data/regenerate.py`, config in
  `sentinel_features/nfstream_config.py`. Source of the copied code: see the
  file headers (FlowFrontiers/CyberML-DataQuality).

## Result

| Label       | Regenerated | Published | Difference      |
| ----------- | ----------- | --------- | --------------- |
| BENIGN      | 457,782     | 456,311   | +1,471 (+0.32%) |
| FTP-Patator | 7,963       | 7,963     | 0%              |
| SSH-Patator | 2,980       | 2,980     | 0%              |
| Total       | 468,725     | 467,254   | +1,471 (+0.31%) |

No NaN labels or rows. Columns are identical to the reference, in the same
order. NFStream produced 571,202 flows; 102,477 single-packet RST/FIN flows were
filtered. **All labels are within 1%: pass.**

## Gap

Regeneration has 1,471 more BENIGN flows (+0.32%). Both attack labels match
exactly, so labelling and attack-flow handling reproduce the published data.
Likely cause: NFStream version or platform differences in how short benign
flows expire. Impact: none on attack labels; BENIGN is well inside the threshold.

## Tests

`tests/test_parity.py`: the 1% rule and column order are tested on small
made-up data (runs in CI). A real-data test runs locally when
`SENTINEL_REGEN_CSV` and `SENTINEL_PUBLISHED_CSV` are set; it passed on
2026-10-29.

## Not covered

Only Tuesday was regenerated. Values per flow and the sensor in file mode
(rest of TC-04) are Stage 2.

## References

- Sharafaldin et al., 2018 (CICIDS2017)
- "Evaluating ML-Based Anomaly Detection Across Datasets of Varied Integrity:
  A Case Study", FlowFrontiers/CyberML-DataQuality

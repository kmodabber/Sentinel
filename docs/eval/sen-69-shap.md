# SEN-69: Global |SHAP| check (R2 leakage guard)

Average |SHAP| across a bounded val sample for the dev stand-in model
(`models/dev-lgbm-noport/`). Purpose: catch the R2 failure mode where
one feature dominates the model (a sign of leakage).

- **Package**: `models/dev-lgbm-noport/` (`dev-lgbm-noport-v0`)
- **Val sample**: 20,000 rows
- **Alerts in sample (threshold 0.98)**: 3,230
- **Verdict**: PASS — no feature carries ≥ 60% of mean |SHAP|. Top share
  is `dst2src_max_ps` at **8.55%**, well below the dominance threshold.

## Top 10 features by mean |SHAP|

| rank | feature                   | mean \|SHAP\| | share |
|------|---------------------------|---------------|-------|
| 1    | dst2src_max_ps            | 3.88          | 8.55% |
| 2    | bidirectional_bytes       | 3.46          | 7.63% |
| 3    | bidirectional_mean_ps     | 2.90          | 6.41% |
| 4    | src2dst_fin_packets       | 2.76          | 6.10% |
| 5    | dst2src_stddev_piat_ms    | 2.25          | 4.96% |
| 6    | src2dst_psh_packets       | 2.13          | 4.70% |
| 7    | bidirectional_max_ps      | 1.75          | 3.86% |
| 8    | bidirectional_rst_packets | 1.69          | 3.74% |
| 9    | dst2src_bytes             | 1.56          | 3.44% |
| 10   | src2dst_max_ps            | 1.50          | 3.30% |

Distribution is diffuse across packet-size and flag-count features — the
shape we want for an NFStream-feature model, and the opposite of the
dst_port-dominated signature that drove the no-port decision.

## Timing (SEN-NFR-01 sanity)

`explain_ms_per_1000_alerts = 1494.61 ms` on this dev box. That's the
per-1000-alert budget for explanation; a single alert costs ~1.5 ms.
Well below the 10 s flow-end-to-alert ceiling.

## Caveats

- Dev stand-in model, not SEN-68. Re-run against the real package before
  shipping: the top-feature share will shift and the R2 guard must pass
  on the real model.
- |SHAP| averaged only over top-5 entries per alert (SEN-FR-12 shape).
  A full 56-feature mean would move numbers but not the dominance test.

## Output

- `results/sen-69/global.json` — full report (top-10, timing, flag).

# SEN-134: Default confidence threshold

Picks the default `alert.min_confidence` the sensor (SEN-75) applies to the
multi-class model's confidence (max class probability), per SEN-FR-18 and
SEN-NFR-04.

**Recommendation: 0.89** (global). This is the highest threshold at which
every learnable class still clears a per-class recall floor of 0.80 while
the overall SEN-NFR-04 targets (attack recall ≥ 0.95, benign FPR ≤ 0.01)
are met with room to spare. `results/sen-134/thresholds.json` carries the
same value per class in the C1 shape; no per-class loosening is needed at
0.89.

> Tuned on the dev stand-in model (`models/dev-lgbm-noport/`), not the
> team SEN-68 package. Re-run `python -m sentinel_ml.threshold_sweep
> --package <real-package>` once SEN-68 lands and update the number.

## Method

- **Input**: `data/processed/val.parquet` scored through
  `sensor.package.score()` so the confidence distribution matches what
  the sensor sees at runtime.
- **Alert rule (SEN-FR-18)**: a flow becomes an alert iff
  `predicted_class ∈ ATTACK_LABELS AND confidence ≥ threshold`.
  Confidence = `max(predict_proba)`.
- **Sweep**: threshold from 0.50 → 0.99 in 0.01 steps
  (`results/sen-134/sweep.csv`).
- **Overall targets (SEN-NFR-04, val)**: attack recall ≥ 0.95 AND benign
  FPR ≤ 0.01.
- **Per-class floor**: every attack class that is learnable at all must
  keep per-class recall ≥ 0.80. Classes that never reach 0.80 anywhere in
  the sweep are excluded and reported (not silently used to drag the
  global threshold down).
- **Choice rule**: among thresholds that satisfy both overall targets and
  the per-class floor over the coverable classes, pick the **highest**.
  Higher threshold = fewer false alarms at equal recall; the per-class
  floor stops a few high-volume classes from hiding slow-DoS / web-BF
  underperformance.

## Table at key thresholds

| threshold | attack recall | benign FPR | alerts / 10k benign |
|-----------|---------------|------------|---------------------|
| 0.50      | 0.9964        | 0.000340   | 3.40                |
| 0.70      | 0.9963        | 0.000133   | 1.33                |
| 0.80      | 0.9942        | 0.000122   | 1.22                |
| **0.89**  | **0.9941**    | **0.000105** | **1.05**          |
| 0.90      | 0.9931        | 0.000101   | 1.01                |
| 0.95      | 0.9911        | 0.000087   | 0.87                |
| 0.98      | 0.9847        | 0.000036   | 0.36                |
| 0.99      | 0.9060        | 0.000036   | 0.36                |

Both overall targets are met from 0.50 through 0.98. The chosen point is
0.89 because above it the per-class floor (see next table) breaks for
DoS Slowhttptest, DoS slowloris and Web Attack - Brute Force.

## Per-class recall at 0.89 (chosen) vs 0.98 (old overall-only pick)

| class                    | recall @ 0.89 | recall @ 0.98 |
|--------------------------|---------------|---------------|
| Bot                      | 1.0000        | 0.9589        |
| DDoS                     | 0.9999        | 0.9999        |
| DoS GoldenEye            | 0.9851        | 0.9768        |
| DoS Hulk                 | 0.9998        | 0.9997        |
| DoS Slowhttptest         | 0.9068        | 0.0763        |
| DoS slowloris            | 1.0000        | 0.3217        |
| FTP-Patator              | 1.0000        | 1.0000        |
| PortScan                 | 0.9885        | 0.9881        |
| SSH-Patator              | 0.9983        | 0.9983        |
| Web Attack - Brute Force | 1.0000        | 0.8571        |
| Rare attack              | 0.3846        | 0.0000        |

Rare attack never reaches the 0.80 floor anywhere in the sweep (peak
0.6154 at 0.70) — the model doesn't learn this merged bucket. It is
excluded from the floor check and listed under
`recommendation.json → never_reach_class_floor`. Flagged for SEN-68
owner / training discussion; raising/lowering threshold won't fix it.

## Caveats

- **Val, not test** (SEN-NFR-06). Live traffic is not val — this is a
  starting point, expect to re-tune after SEN-135 and once we have real
  capture data.
- **Dev stand-in model** — the real SEN-68 package will have different
  probability calibration. Re-run the sweep against the real package
  before shipping.
- The chart (attack recall & benign FPR vs. threshold) is pending —
  `matplotlib` is not in the current env; `results/sen-134/sweep.csv`
  has the raw data if you want to render one locally.

## Outputs

- `results/sen-134/sweep.csv` — full 50-row sweep (per-class recall columns)
- `results/sen-134/recommendation.json` — chosen threshold + basis metrics
  + `class_floor` + `never_reach_class_floor`
- `results/sen-134/thresholds.json` — C1-shaped per-class map for the package

```json
{
  "Bot": 0.89, "DDoS": 0.89, "DoS GoldenEye": 0.89, "DoS Hulk": 0.89,
  "DoS Slowhttptest": 0.89, "DoS slowloris": 0.89, "FTP-Patator": 0.89,
  "PortScan": 0.89, "Rare attack": 0.89, "SSH-Patator": 0.89,
  "Web Attack - Brute Force": 0.89
}
```

- `docs/eval/sen-134-note-for-arshan.md` — short note for SEN-75 wiring

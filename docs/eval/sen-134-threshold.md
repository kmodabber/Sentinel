# SEN-134: Default confidence threshold

Picks the default `alert.min_confidence` the sensor (SEN-75) applies to the
multi-class model's confidence (max class probability), per SEN-FR-18 and
SEN-NFR-04.

**Recommendation: 0.98** (global). Per-class overrides for three
underperforming classes are written to `results/sen-134/thresholds.json`
in the C1 shape (DoS Slowhttptest, DoS slowloris, Web Attack - Brute Force
drop to 0.5 so the class is not silently suppressed; see caveats).

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
- **Targets (SEN-NFR-04, val)**: attack recall ≥ 0.95 AND benign FPR ≤ 0.01.
- **Choice rule**: pick the **highest** threshold that still meets both
  targets. Higher threshold = fewer false alarms at equal recall, so
  being greedy on threshold is the right trade-off.

## Table at key thresholds

| threshold | attack recall | benign FPR | alerts / 10k benign |
|-----------|---------------|------------|---------------------|
| 0.50      | 0.9964        | 0.000340   | 3.40                |
| 0.70      | 0.9963        | 0.000133   | 1.33                |
| 0.80      | 0.9942        | 0.000122   | 1.22                |
| 0.90      | 0.9931        | 0.000101   | 1.01                |
| 0.95      | 0.9911        | 0.000087   | 0.87                |
| **0.98**  | **0.9847**    | **0.000036** | **0.36**          |
| 0.99      | 0.9060        | 0.000036   | 0.36                |

Both targets are met from 0.50 through 0.98. At 0.99 attack recall drops
below the 0.95 target, so 0.98 is the chosen global default.

## Per-class recall at 0.98

| class                   | recall |
|-------------------------|--------|
| Bot                     | 0.9589 |
| DDoS                    | 0.9999 |
| DoS GoldenEye           | 0.9768 |
| DoS Hulk                | 0.9997 |
| DoS Slowhttptest        | 0.0763 |
| DoS slowloris           | 0.3217 |
| FTP-Patator             | 1.0000 |
| PortScan                | 0.9881 |
| Rare attack             | 0.0000 |
| SSH-Patator             | 0.9983 |
| Web Attack - Brute Force| 0.8571 |

Three classes fall far below the global target even at 0.5:
DoS Slowhttptest, DoS slowloris, Web Attack - Brute Force. For these,
`thresholds.json` carries a lowered per-class threshold (0.5) so the
sensor can still surface those alerts if the package owner wires the
per-class override into `score()`.

Rare attack is 0 across the sweep — the model does not learn this
label; raising/lowering threshold won't help. Flagged for SEN-68
owner / training discussion.

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
- `results/sen-134/thresholds.json` — C1-shaped per-class map for the package
- `docs/eval/sen-134-note-for-arshan.md` — short note for SEN-75 wiring

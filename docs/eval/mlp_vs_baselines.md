# MLP vs. baselines on NFS-2023-TE (SEN-FR-13)

- Data version: `nfs2023te-v1`  |  Code commit: `2e04372`  |  Seed: 42
- Test set evaluated **once** per model, same 12-class contract as SEN-109 baselines.
- Training: PyTorch MLP (128-64, ReLU, dropout 0.2, Adam, CE with class weights fit on train only; early stopping on val macro-F1).
- Binary: train_sample=full, best_epoch(with_port)=18, best_epoch(no_port)=20, train_time=5839.7s (both variants)
- Multiclass: train_sample=full, best_epoch(with_port)=13, best_epoch(no_port)=18, train_time=4998.3s (both variants)

## Headline comparison

### Binary — with dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.4556 | 0.0000 | 0.8369 |
| logreg | 0.9645 | 0.0140 | 0.9805 |
| **MLP (SEN-FR-13)** | 0.9940 | 0.0002 | 0.9967 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Binary — without dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.4556 | 0.0000 | 0.8369 |
| logreg | 0.9650 | 0.0136 | 0.9807 |
| **MLP (SEN-FR-13)** | 0.9924 | 0.0006 | 0.9959 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Multiclass — with dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.0759 | 0.0000 | 0.8369 |
| logreg | 0.5569 | 0.0370 | 0.9648 |
| **MLP (SEN-FR-13)** | 0.7133 | 0.0165 | 0.9821 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Multiclass — without dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.0759 | 0.0000 | 0.8369 |
| logreg | 0.5140 | 0.0623 | 0.9435 |
| **MLP (SEN-FR-13)** | 0.6388 | 0.0198 | 0.9796 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

## Per-class recall, multiclass (dst_port on)

| class | support | logreg recall | MLP recall | Δ (MLP − logreg) |
|---|---:|---:|---:|---:|
| BENIGN | 527384 | 0.9630 | 0.9835 | +0.0205 |
| Bot | 148 | 0.9932 | 0.9932 | +0.0000 |
| DDoS | 19139 | 0.9987 | 0.9982 | -0.0005 |
| DoS GoldenEye | 1585 | 0.9249 | 0.9211 | -0.0038 |
| DoS Hulk | 31711 | 0.9994 | 0.9958 | -0.0036 |
| DoS Slowhttptest | 594 | 0.5354 | 0.9327 | +0.3973 |
| DoS slowloris | 1039 | 0.7575 | 0.7228 | -0.0347 |
| FTP-Patator | 1593 | 1.0000 | 1.0000 | +0.0000 |
| PortScan | 46343 | 0.9581 | 0.9590 | +0.0009 |
| Rare attack | 18 | 0.5556 | 0.5000 | -0.0556 |
| SSH-Patator | 596 | 0.9178 | 0.9530 | +0.0352 |
| Web Attack - Brute Force | 31 | 1.0000 | 1.0000 | +0.0000 |

## With vs. without dst_port (MLP)

- binary: Δ macro-F1 = +0.0016, Δ benign FPR = -0.0004 (with − without dst_port)
- multiclass: Δ macro-F1 = +0.0745, Δ benign FPR = -0.0033 (with − without dst_port)

## Top features (permutation importance, MLP)

| task | dst_port | top-5 features (importance) |
|---|---|---|
| binary | on  | bidirectional_mean_ps (0.096), bidirectional_max_ps (0.091), src2dst_max_ps (0.064), bidirectional_stddev_ps (0.060), **dst_port (0.033)** |
| binary | off | src2dst_max_ps (0.156), bidirectional_mean_ps (0.102), src2dst_ack_packets (0.077), bidirectional_max_ps (0.051), dst2src_packets (0.049) |
| multiclass | on  | bidirectional_mean_ps (0.261), **dst_port (0.229)**, src2dst_max_ps (0.224), src2dst_mean_ps (0.166), dst2src_psh_packets (0.149) |
| multiclass | off | src2dst_max_ps (0.239), bidirectional_mean_ps (0.199), src2dst_mean_ps (0.183), bidirectional_min_ps (0.126), src2dst_min_ps (0.118) |

## Leakage flags

Automatic thresholds (from `_leakage_warning`, train_mlp.py):

- MLP **val macro-F1 for binary** is 0.9981 (with_port) / 0.9973 (no_port) — just below the 0.999 "near-perfect" alarm but close enough to warrant the investigation below.
- MLP vs. logreg margin on binary is +0.0295 (port on) / +0.0274 (port off) — inside the 0.01 "MLP learns nothing extra" margin threshold *for logreg*, i.e. **logreg is only ~3pt behind the MLP on binary**. That is the signal doc 03 asks us to flag.
- No alarm triggered on multiclass: MLP − logreg is +0.156 (port on) / +0.125 (port off).

Baselines-side leakage flags carried forward (SEN-109):
- WARNING  logreg/binary/port=True: macro_f1 = 0.9645 ≥ 0.95
- WARNING  logreg/binary/port=False: macro_f1 = 0.9650 ≥ 0.95

## Leakage investigation & remediation

Two separate suspicions, with different roots:

### 1. Binary is _too easy_ (both logreg and MLP score ≥0.96 with port off)

- Removing `dst_port` costs the MLP only 0.0016 macro-F1 on binary, and the logreg table is nearly identical. The ports are **not** the leak on the binary task.
- The permutation ranking is dominated by **packet-size statistics** (`bidirectional_mean_ps`, `src2dst_max_ps`, `bidirectional_stddev_ps`). The NFS-2023-TE capture includes long benign TLS/HTTPS flows alongside short attack probe flows (PortScan, Patator). A linear separator on packet-size mean / max already finds them → logreg ≥ 0.96.
- **Why this looks like leakage**: the attacks in CIC-IDS-2017 were generated by a small set of tools (slowhttptest, hping3, hulk, …) run against a specific target host for a specific capture window. The *tool* leaves a near-constant packet-size fingerprint per attack class, and because the split is time-based with the attacks clustered in specific windows, train/test share the same tool fingerprint instead of "same attack family, different time".
- **Fixes** (in order of cost):
  1. **Split rework (SEN-106 / Mikhai)** — move to a per-flow-group or per-capture-session split so the same tool run doesn't appear on both sides. Doc 03 explicitly makes this Mikhai's work, not ours. Do **not** change the split from the MLP side; it would make the comparison with baselines meaningless.
  2. **Feature audit** — drop or clip the packet-size stats that are nearly constant inside an attack class; bucketize `bidirectional_max_ps` and `bidirectional_stddev_ps`; add ratio-based features that are tool-independent (e.g., `bytes_per_packet`, inter-arrival-time quantiles).
  3. **Harder evaluation protocol** — add a *leave-one-attack-day-out* or *leave-one-tool-out* eval on top of the existing split so we can see what the model really generalises to. This is a report-only change (SEN-FR-09 eval template already supports it in principle); no model retraining required.

### 2. `dst_port` is a real leak for multiclass (but not for binary)

- With dst_port **on**, multiclass MLP macro-F1 = 0.7133; with it **off**, 0.6388. A **0.0745 drop** is far above the 0.01 threshold.
- `dst_port` is the #2 permutation-importance feature on multiclass (0.229) — well above the next non-packet-size feature.
- This is the FTP/SSH-Patator / SlowHTTPTest / PortScan tell: each attack class in CIC-IDS-2017 was run against a specific target port, so the model learns `port=21 → FTP-Patator`, `port=22 → SSH-Patator`, etc. That is a dataset artefact, not a generalisable signal — in production the attacker chooses the port.
- **Fixes**:
  1. **Report both with- and without-port numbers** (already doing this; doc 03 required it). The _without-port_ number is the honest one for the "does the model see attack patterns, or does it memorise target ports?" question.
  2. **Deploy the no-port model** on the sensor. The with-port model is fine for CTS-demo comparison tables, but FR-01/FR-02 inference should use the dst_port-off preprocessor so port memorisation can't mask a real detection failure.
  3. **Augmentation**: shuffle `dst_port` within a class in training so the model has to rely on other features. Cheap, worth trying once SEN-106 lands.
  4. **Replace `dst_port` with a port-category feature** (well-known / registered / ephemeral) during feature engineering — keeps a useful signal, kills the exact-port memorisation.

### Not acting on this now, intentionally

Per doc 03, any "fix the split" work belongs to SEN-106. The right next step is to merge this PR with the leakage flagged, then let SEN-106 produce a cleaner split and re-run `python -m sentinel_ml.train_mlp --task both` plus `python scripts/build_mlp_vs_baselines.py` — no code changes needed on the SEN-FR-13 side.

## Did deep learning help?

- **binary**: MLP macro-F1 0.9940 beats logreg 0.9645 (Δ +0.0295).
- **multiclass**: MLP macro-F1 0.7133 beats logreg 0.5569 (Δ +0.1564).

Honest read: on this dataset with time-based splits, the gap between logreg and the small MLP is the real answer to "is deep learning worth it here?". Where the MLP gap is small (≤0.01), a well-regularised linear model is enough and is cheaper to ship on the sensor. Where the gap is larger — typically on the minority DoS/Web-Attack classes — the MLP earns its place in the evaluation. Tree-based models (SEN-FR-08) still need to be filled in before a final verdict; this file updates automatically once their eval JSONs land.

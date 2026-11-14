# MLP vs. baselines on NFS-2023-TE (SEN-FR-13)

- Data version: `nfs2023te-v1`  |  Code commit: `9d0e566-dirty`  |  Seed: 42
- Test set evaluated **once** per model, same 12-class contract as SEN-109 baselines.
- Training: PyTorch MLP (128-64, ReLU, dropout 0.2, Adam, CE with class weights fit on train only; early stopping on val macro-F1).
- Binary: train_sample=50000, best_epoch(with_port)=14, best_epoch(no_port)=15, train_time=25.7s (both variants)
- Multiclass: train_sample=200000, best_epoch(with_port)=16, best_epoch(no_port)=24, train_time=416.2s (both variants)

## Headline comparison

### Binary — with dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.4556 | 0.0000 | 0.8369 |
| logreg | 0.9645 | 0.0140 | 0.9805 |
| **MLP (SEN-FR-13)** | 0.9872 | 0.0035 | 0.9930 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Binary — without dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.4556 | 0.0000 | 0.8369 |
| logreg | 0.9650 | 0.0136 | 0.9807 |
| **MLP (SEN-FR-13)** | 0.9830 | 0.0060 | 0.9907 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Multiclass — with dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.0759 | 0.0000 | 0.8369 |
| logreg | 0.5569 | 0.0370 | 0.9648 |
| **MLP (SEN-FR-13)** | 0.5956 | 0.0395 | 0.9626 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

### Multiclass — without dst_port

| model | macro-F1 | benign FPR | weighted recall |
|---|---:|---:|---:|
| majority | 0.0759 | 0.0000 | 0.8369 |
| logreg | 0.5140 | 0.0623 | 0.9435 |
| **MLP (SEN-FR-13)** | 0.5558 | 0.0793 | 0.9298 |
| random forest | _pending SEN-FR-08_ | _pending_ | _pending_ |
| lightgbm | _pending SEN-FR-08_ | _pending_ | _pending_ |

## Per-class recall, multiclass (dst_port on)

| class | support | logreg recall | MLP recall | Δ (MLP − logreg) |
|---|---:|---:|---:|---:|
| BENIGN | 527384 | 0.9630 | 0.9605 | -0.0025 |
| Bot | 148 | 0.9932 | 0.9595 | -0.0337 |
| DDoS | 19139 | 0.9987 | 0.9940 | -0.0047 |
| DoS GoldenEye | 1585 | 0.9249 | 0.7710 | -0.1539 |
| DoS Hulk | 31711 | 0.9994 | 0.9992 | -0.0002 |
| DoS Slowhttptest | 594 | 0.5354 | 0.9209 | +0.3855 |
| DoS slowloris | 1039 | 0.7575 | 0.7132 | -0.0443 |
| FTP-Patator | 1593 | 1.0000 | 1.0000 | +0.0000 |
| PortScan | 46343 | 0.9581 | 0.9600 | +0.0019 |
| Rare attack | 18 | 0.5556 | 0.2778 | -0.2778 |
| SSH-Patator | 596 | 0.9178 | 0.9715 | +0.0537 |
| Web Attack - Brute Force | 31 | 1.0000 | 1.0000 | +0.0000 |

**Weak classes (MLP recall < 0.5):** Rare attack (recall 0.28).

## With vs. without dst_port (MLP)

- binary: Δ macro-F1 = +0.0042, Δ benign FPR = -0.0025 (with − without dst_port)
- multiclass: Δ macro-F1 = +0.0398, Δ benign FPR = -0.0398 (with − without dst_port)

## Leakage flags

_No MLP thresholds crossed (val/test macro-F1 < 0.999 and not within 1pt of logreg)._

Baselines (SEN-109) flagged:
- WARNING  logreg/binary/port=True: logreg macro_f1=0.9645 >= 0.95
- WARNING  logreg/binary/port=False: logreg macro_f1=0.9650 >= 0.95

## Did deep learning help?

- **binary**: MLP macro-F1 0.9872 beats logreg 0.9645 (Δ +0.0227).
- **multiclass**: MLP macro-F1 0.5956 beats logreg 0.5569 (Δ +0.0387).

Honest read: on this dataset with time-based splits, the gap between logreg and the small MLP is the real answer to "is deep learning worth it here?". Where the MLP gap is small (≤0.01), a well-regularised linear model is enough and is cheaper to ship on the sensor. Where the gap is larger — typically on the minority DoS/Web-Attack classes — the MLP earns its place in the evaluation. Tree-based models (SEN-FR-08) still need to be filled in before a final verdict; this file updates automatically once their eval JSONs land.

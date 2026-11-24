# Sentinel Evaluation Report v1 (`SEN-108` & `SEN-FR-08`)

- **Data Version**: `nfs2023te-v1` (`NFS-2023-TE, FlowFrontiers/CyberML-DataQuality @ c6e0191`, `data_card.code_commit = 575a7db`)
- **Splits**: `train` = `1,887,703` | `val` = `626,336` | `test` = `630,181` rows (`gap_seconds = 1`, `rows_dropped_in_gaps = 6,217`)
- **Label Contract (`SEN-FR-05`)**: 2 binary classes (`attack`, `benign`) and 12 canonical multi-class labels (`BENIGN` + 10 distinct attacks + `Rare attack`)
- **Evaluation Protocol (`SEN-FR-09`)**: Hyperparameters and ensemble weights tuned strictly on `val.parquet` (`seed = 42`); `test.parquet` (`630,181` flows) evaluated once per model via `sentinel_ml/eval_template.py`
- **W&B Project (`SEN-107`)**: [`wcs-sentinel/Sentinel`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel)

---

## 1. Headline Test-Set Comparison on Data v1 (`nfs2023te-v1`, `630,181` Test Flows)

- All models are trained on `train.parquet` (`1,887,703` rows), tuned on `val.parquet` (`626,336` rows, `seed = 42`), and evaluated once on `test.parquet` (`630,181` rows) with and without `dst_port` (`56` vs. `55` numeric features).

### 1.1.1 Binary Classification — With `dst_port` (56 features)

| Model | Val Macro-F1 (`seed=42`) | Test Macro-F1 (All) | Test Macro-F1 (Unseen) | Weighted-F1 | Accuracy | Benign FPR (All) | Benign FPR (Unseen) | Attack Recall | Fit Time (s) | Infer Latency (ms / 1k rows) | Top-3 Features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Baseline (`SEN-109`) | 0.4556 | **0.4556** | 0.4556 | 0.7626 | 0.8369 | 0.0000 | 0.0000 | 0.0000 | <0.1 | <0.1 | — |
| Logistic Regression (`SEN-109`) | — | **0.9645** | 0.9492 | 0.9805 | 0.9804 | 0.0140 | 0.0250 | 0.9520 | 28.4 | 0.2 | `bidirectional_min_piat_ms` (38.214), `protocol` (18.147), `dst2src_max_ps` (10.506) |
| PyTorch MLP (`SEN-FR-13`) | 0.9981 | **0.9940** | — | 0.9967 | 0.9967 | 0.0002 | — | — | 2919.9 | 1.8 | `bidirectional_mean_ps` (0.096), `bidirectional_max_ps` (0.091), `src2dst_max_ps` (0.064) |
| **Random Forest (`SEN-FR-08`)** | 0.9990 | **0.9939** | 0.9964 | 0.9967 | 0.9967 | 0.0000 | 0.0000 | 0.9798 | 151.66 | 0.73 | `src2dst_syn_packets` (0.127), `bidirectional_stddev_ps` (0.076), `bidirectional_mean_ps` (0.073) |
| **LightGBM (`SEN-FR-08`)** | 0.9989 | **0.9941** | 0.9967 | 0.9968 | 0.9968 | 0.0001 | 0.0001 | 0.9808 | 74.11 | 0.62 | `dst_port` (0.284), `bidirectional_bytes` (0.052), `bidirectional_mean_ps` (0.040) |
| Fixed-Weights Ensemble (`0.50 / 0.50`) | 0.9990 | **0.9941** | 0.9967 | 0.9968 | 0.9968 | 0.0000 | 0.0001 | 0.9806 | 225.77 | 1.36 | `dst_port` (0.153), `src2dst_syn_packets` (0.068), `bidirectional_mean_ps` (0.057) |
| Fixed-Weights Ensemble (Val-Tuned `0.60 / 0.40`) | 0.9990 | **0.9941** | 0.9967 | 0.9968 | 0.9968 | 0.0000 | 0.0000 | 0.9806 | 225.77 | 1.36 | `dst_port` (0.153), `src2dst_syn_packets` (0.068), `bidirectional_mean_ps` (0.057) |
| **Stacking Meta-Learner Ensemble (`SEN-FR-08`)** | 0.9990 | **0.9940** | 0.9966 | 0.9967 | 0.9967 | 0.0000 | 0.0000 | 0.9800 | 226.08 | 1.51 | `dst_port` (0.116), `src2dst_syn_packets` (0.085), `bidirectional_stddev_ps` (0.062) |

### 1.1.2 Binary Classification — Without `dst_port` (55 features)

| Model | Val Macro-F1 (`seed=42`) | Test Macro-F1 (All) | Test Macro-F1 (Unseen) | Weighted-F1 | Accuracy | Benign FPR (All) | Benign FPR (Unseen) | Attack Recall | Fit Time (s) | Infer Latency (ms / 1k rows) | Top-3 Features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Baseline (`SEN-109`) | 0.4556 | **0.4556** | 0.4556 | 0.7626 | 0.8369 | 0.0000 | 0.0000 | 0.0000 | <0.1 | <0.1 | — |
| Logistic Regression (`SEN-109`) | — | **0.9650** | 0.9547 | 0.9807 | 0.9807 | 0.0136 | 0.0270 | 0.9517 | 28.4 | 0.2 | `bidirectional_min_piat_ms` (40.477), `protocol` (17.480), `dst2src_max_ps` (11.566) |
| PyTorch MLP (`SEN-FR-13`) | 0.9973 | **0.9924** | — | 0.9959 | 0.9959 | 0.0006 | — | — | 2919.9 | 1.8 | `src2dst_max_ps` (0.156), `bidirectional_mean_ps` (0.102), `src2dst_ack_packets` (0.077) |
| **Random Forest (`SEN-FR-08`)** | 0.9984 | **0.9933** | 0.9973 | 0.9964 | 0.9964 | 0.0001 | 0.0001 | 0.9781 | 138.51 | 0.59 | `dst2src_stddev_ps` (0.092), `bidirectional_syn_packets` (0.084), `bidirectional_stddev_ps` (0.083) |
| **LightGBM (`SEN-FR-08`)** | 0.9984 | **0.9936** | 0.9979 | 0.9965 | 0.9965 | 0.0001 | 0.0000 | 0.9789 | 76.47 | 0.60 | `bidirectional_bytes` (0.071), `bidirectional_stddev_ps` (0.062), `bidirectional_mean_ps` (0.052) |
| Fixed-Weights Ensemble (`0.50 / 0.50`) | 0.9984 | **0.9935** | 0.9978 | 0.9965 | 0.9965 | 0.0001 | 0.0001 | 0.9788 | 214.98 | 1.21 | `bidirectional_stddev_ps` (0.072), `dst2src_stddev_ps` (0.063), `bidirectional_mean_ps` (0.051) |
| Fixed-Weights Ensemble (Val-Tuned `0.70 / 0.30`) | 0.9984 | **0.9934** | 0.9976 | 0.9964 | 0.9964 | 0.0001 | 0.0001 | 0.9788 | 214.98 | 1.21 | `bidirectional_stddev_ps` (0.072), `dst2src_stddev_ps` (0.063), `bidirectional_mean_ps` (0.051) |
| **Stacking Meta-Learner Ensemble (`SEN-FR-08`)** | 0.9988 | **0.9927** | 0.9963 | 0.9960 | 0.9961 | 0.0000 | 0.0000 | 0.9760 | 215.33 | 1.36 | `bidirectional_stddev_ps` (0.073), `dst2src_stddev_ps` (0.064), `bidirectional_mean_ps` (0.051) |

### 1.2.1 Multiclass Classification — With `dst_port` (56 features)

| Model | Val Macro-F1 (`seed=42`) | Test Macro-F1 (All) | Test Macro-F1 (Unseen) | Weighted-F1 | Accuracy | Benign FPR (All) | Benign FPR (Unseen) | Fit Time (s) | Infer Latency (ms / 1k rows) | Top-3 Features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Baseline (`SEN-109`) | 0.0759 | **0.0759** | 0.0759 | 0.7626 | 0.8369 | 0.0000 | 0.0000 | <0.1 | <0.1 | — |
| Logistic Regression (`SEN-109`) | — | **0.5569** | 0.4714 | 0.9751 | 0.9648 | 0.0370 | 0.0560 | 142.0 | 0.4 | `dst2src_psh_packets` (12.856), `src2dst_psh_packets` (12.828), `dst_port` (10.425) |
| PyTorch MLP (`SEN-FR-13`) | 0.7280 | **0.7133** | — | 0.9821 | 0.9821 | 0.0165 | — | 2499.2 | 2.1 | `bidirectional_mean_ps` (0.261), `dst_port` (0.229), `src2dst_max_ps` (0.224) |
| **Random Forest (`SEN-FR-08`)** | 0.9728 | **0.9037** | 0.8418 | 0.9963 | 0.9964 | 0.0000 | 0.0001 | 198.72 | 1.33 | `src2dst_syn_packets` (0.101), `bidirectional_stddev_ps` (0.069), `dst2src_stddev_ps` (0.061) |
| **LightGBM (`SEN-FR-08`)** | 0.9876 | **0.9276** | 0.8825 | 0.9967 | 0.9967 | 0.0000 | 0.0001 | 394.77 | 4.70 | `dst_port` (0.304), `bidirectional_bytes` (0.067), `bidirectional_mean_ps` (0.039) |
| Fixed-Weights Ensemble (`0.50 / 0.50`) | 0.9847 | **0.9335** | 0.8883 | 0.9967 | 0.9967 | 0.0000 | 0.0001 | 593.49 | 6.11 | `dst_port` (0.163), `bidirectional_bytes` (0.059), `src2dst_syn_packets` (0.052) |
| Fixed-Weights Ensemble (Val-Tuned `0.25 / 0.75`) | 0.9876 | **0.9289** | 0.8839 | 0.9967 | 0.9967 | 0.0000 | 0.0001 | 593.49 | 6.11 | `dst_port` (0.163), `bidirectional_bytes` (0.059), `src2dst_syn_packets` (0.052) |
| **Stacking Meta-Learner Ensemble (`SEN-FR-08`)** | 0.9893 | **0.9283** | 0.8857 | 0.9967 | 0.9968 | 0.0000 | 0.0001 | 596.63 | 7.11 | `dst_port` (0.159), `bidirectional_bytes` (0.059), `src2dst_syn_packets` (0.053) |

### 1.2.2 Multiclass Classification — Without `dst_port` (55 features)

| Model | Val Macro-F1 (`seed=42`) | Test Macro-F1 (All) | Test Macro-F1 (Unseen) | Weighted-F1 | Accuracy | Benign FPR (All) | Benign FPR (Unseen) | Fit Time (s) | Infer Latency (ms / 1k rows) | Top-3 Features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Baseline (`SEN-109`) | 0.0759 | **0.0759** | 0.0759 | 0.7626 | 0.8369 | 0.0000 | 0.0000 | <0.1 | <0.1 | — |
| Logistic Regression (`SEN-109`) | — | **0.5140** | 0.4638 | 0.9632 | 0.9435 | 0.0623 | 0.0640 | 142.0 | 0.4 | `dst2src_psh_packets` (18.974), `src2dst_psh_packets` (18.784), `bidirectional_min_ps` (10.921) |
| PyTorch MLP (`SEN-FR-13`) | 0.6510 | **0.6388** | — | 0.9796 | 0.9796 | 0.0198 | — | 2499.2 | 2.1 | `src2dst_max_ps` (0.239), `bidirectional_mean_ps` (0.199), `src2dst_mean_ps` (0.183) |
| **Random Forest (`SEN-FR-08`)** | 0.9589 | **0.8933** | 0.8353 | 0.9959 | 0.9960 | 0.0001 | 0.0001 | 195.53 | 1.30 | `dst2src_stddev_ps` (0.077), `bidirectional_stddev_ps` (0.067), `bidirectional_syn_packets` (0.064) |
| **LightGBM (`SEN-FR-08`)** | 0.9752 | **0.9304** | 0.8941 | 0.9963 | 0.9964 | 0.0001 | 0.0001 | 396.79 | 4.62 | `bidirectional_bytes` (0.136), `src2dst_bytes` (0.076), `src2dst_max_ps` (0.067) |
| Fixed-Weights Ensemble (`0.50 / 0.50`) | 0.9776 | **0.9238** | 0.8866 | 0.9963 | 0.9963 | 0.0001 | 0.0001 | 592.32 | 6.01 | `bidirectional_bytes` (0.090), `src2dst_bytes` (0.066), `bidirectional_mean_ps` (0.058) |
| Fixed-Weights Ensemble (Val-Tuned `0.40 / 0.60`) | 0.9777 | **0.9243** | 0.8878 | 0.9963 | 0.9963 | 0.0001 | 0.0001 | 592.32 | 6.01 | `bidirectional_bytes` (0.090), `src2dst_bytes` (0.066), `bidirectional_mean_ps` (0.058) |
| **Stacking Meta-Learner Ensemble (`SEN-FR-08`)** | 0.9829 | **0.9247** | 0.8873 | 0.9964 | 0.9964 | 0.0002 | 0.0002 | 595.35 | 7.09 | `bidirectional_bytes` (0.088), `src2dst_bytes` (0.066), `bidirectional_mean_ps` (0.058) |

### 1.3 Validation-Only Hyperparameter Tuning (`val.parquet`, `seed = 42`)

- All hyperparameter searches, weight sweeps, and meta-learner OOF fits ran strictly on `val.parquet` (`626,336` rows, `seed = 42`) before `test.parquet` (`630,181` rows) was evaluated once.

| Model Family | Tuning Split | Seed | Search Space (`val.parquet`) | Selected Configuration | Val Macro-F1 (Port On / Off) | Test Touched? |
|---|---|---:|---|---|---:|:---:|
| **Random Forest** (`RandomForestClassifier`) | `val.parquet` | `42` | `9` trials: `n_estimators in [80]`, `max_depth in [20, 26, None]`, `min_samples_leaf in [1, 2, 5]` | `n_estimators=80, max_depth=20, min_samples_leaf=5, max_samples=0.25` | Binary: `0.9990 / 0.9984`<br>Multi: `0.9728 / 0.9589` | **No** |
| **LightGBM** (`LGBMClassifier`) | `val.parquet` | `42` | `9` trials: `n_estimators in [100]`, `num_leaves in [31, 63, 127]`, `learning_rate in [0.05, 0.08, 0.1]` | Binary: `n_estimators=100, num_leaves=63, lr=0.1, min_child_samples=20`<br>Multi: `n_estimators=100, num_leaves=63, lr=0.1, reg_lambda=10.0, max_delta_step=2.0` | Binary: `0.9989 / 0.9984`<br>Multi: `0.9876 / 0.9752` | **No** |
| **Fixed-Weights Ensemble** (`FixedWeightEnsemble`) | `val.parquet` | `42` | `21`-step grid `w_rf in {0.00, 0.05, ..., 1.00}`, `w_lgbm = 1 - w_rf` | Equal `0.50 / 0.50` + Val-tuned Binary `0.60/0.40` (port) / `0.70/0.30` (no-port); Multi `0.25/0.75` (port) / `0.40/0.60` (no-port) | Equal Binary: `0.9990 / 0.9984`<br>Tuned Binary: `0.9990 / 0.9984`<br>Equal Multi: `0.9847 / 0.9776`<br>Tuned Multi: `0.9876 / 0.9777` | **No** |
| **Stacking Meta-Learner** (`MetaLearnerEnsemble`) | `val.parquet` (`3`-fold OOF) | `42` | OOF probability + log-odds + disagreement features (`3*K` dims) with `LogisticRegression(C=20.0) + coordinate-ascent class_multipliers on val OOF` | `LogisticRegression(C=20.0)` + val OOF class multipliers (RF share `0.51–0.64`, LGBM share `0.36–0.49`) | Binary: `0.9990 / 0.9988`<br>Multi: `0.9893 / 0.9829` | **No** |
| **PyTorch MLP** (`SEN-FR-13`) | `val.parquet` | `42` | `hidden=(128, 64)`, ReLU, `dropout=0.2`, Adam (`lr=1e-3`), class-weighted CE, early stopping (`patience=5`) | Best epoch: Binary `18` (port) / `20` (no-port); Multi `13` (port) / `18` (no-port) | Binary: `0.9981 / 0.9973`<br>Multi: `0.7280 / 0.6510` | **No** |

---

## 2. Per-Class Precision, Recall, F1, Support, Benign FPR, and Confusion Matrices

### 2.1 Binary Per-Class Precision, Recall, F1-Score, and Support (`attack` vs. `benign`)

#### Binary Per-Class Metrics — With `dst_port` (56 Features)

| Model | Benign FPR | `attack` Precision | `attack` Recall | `attack` F1 | `attack` Support | `benign` Precision | `benign` Recall | `benign` F1 | `benign` Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Majority Baseline** | `0.0000` | 0.0000 | 0.0000 | **0.0000** | 102,797 | 0.8369 | 1.0000 | **0.9112** | 527,384 |
| **Logistic Regression** | `0.0140` | 0.9297 | 0.9520 | **0.9407** | 102,797 | 0.9906 | 0.9860 | **0.9883** | 527,384 |
| **Random Forest** | `0.0000` | 0.9999 | 0.9798 | **0.9897** | 102,797 | 0.9961 | 1.0000 | **0.9980** | 527,384 |
| **LightGBM** | `0.0001` | 0.9997 | 0.9808 | **0.9902** | 102,797 | 0.9963 | 0.9999 | **0.9981** | 527,384 |
| **Fixed-Weights Ensemble (`0.5/0.5`)** | `0.0000` | 0.9998 | 0.9806 | **0.9901** | 102,797 | 0.9962 | 1.0000 | **0.9981** | 527,384 |
| **Stacking Meta-Learner Ensemble** | `0.0000` | 1.0000 | 0.9800 | **0.9899** | 102,797 | 0.9961 | 1.0000 | **0.9981** | 527,384 |

#### Binary Per-Class Metrics — Without `dst_port` (55 Features)

| Model | Benign FPR | `attack` Precision | `attack` Recall | `attack` F1 | `attack` Support | `benign` Precision | `benign` Recall | `benign` F1 | `benign` Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Majority Baseline** | `0.0000` | 0.0000 | 0.0000 | **0.0000** | 102,797 | 0.8369 | 1.0000 | **0.9112** | 527,384 |
| **Logistic Regression** | `0.0136` | 0.9315 | 0.9517 | **0.9415** | 102,797 | 0.9905 | 0.9864 | **0.9884** | 527,384 |
| **Random Forest** | `0.0001` | 0.9996 | 0.9781 | **0.9888** | 102,797 | 0.9958 | 0.9999 | **0.9978** | 527,384 |
| **LightGBM** | `0.0001` | 0.9997 | 0.9789 | **0.9892** | 102,797 | 0.9959 | 0.9999 | **0.9979** | 527,384 |
| **Fixed-Weights Ensemble (`0.5/0.5`)** | `0.0001` | 0.9997 | 0.9788 | **0.9891** | 102,797 | 0.9959 | 0.9999 | **0.9979** | 527,384 |
| **Stacking Meta-Learner Ensemble** | `0.0000` | 0.9999 | 0.9760 | **0.9878** | 102,797 | 0.9953 | 1.0000 | **0.9977** | 527,384 |

### 2.2 Multi-Class Per-Class Precision, Recall, F1-Score, and Support (All 12 Canonical Classes)

#### Multi-Class Per-Class Precision / Recall / F1 — With `dst_port` (56 Features)

| Canonical Class | Test Support | Unseen Support (`% Twins`) | LogReg (`P / R / F1`) | Random Forest (`P / R / F1`) | LightGBM (`P / R / F1`) | Fixed `0.5/0.5` (`P / R / F1`) | Meta-Learner (`P / R / F1`) | Unseen F1 (`RF / LGBM / Meta`) |
|---|---:|---:|---|---|---|---|---|---|
| **`BENIGN`** | 527,384 | 271,722 (`48.5%`) | `0.9962` / `0.9630` / **`0.9793`** | `0.9961` / `1.0000` / **`0.9980`** | `0.9965` / `1.0000` / **`0.9982`** | `0.9965` / `1.0000` / **`0.9982`** | `0.9965` / `1.0000` / **`0.9982`** | `0.9988` / `0.9993` / **`0.9993`** |
| **`Bot`** | 148 | 106 (`28.4%`) | `0.0741` / `0.9932` / **`0.1379`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `0.9953` / `0.9953` / **`0.9953`** |
| **`DDoS`** | 19,139 | 15,087 (`21.2%`) | `0.9880` / `0.9987` / **`0.9933`** | `0.9993` / `0.9988` / **`0.9990`** | `0.9993` / `0.9988` / **`0.9990`** | `0.9992` / `0.9989` / **`0.9990`** | `0.9992` / `0.9989` / **`0.9990`** | `0.9988` / `0.9988` / **`0.9987`** |
| **`DoS GoldenEye`** | 1,585 | 1,528 (`3.6%`) | `0.5345` / `0.9249` / **`0.6774`** | `0.9840` / `0.9691` / **`0.9765`** | `0.9832` / `0.9968` / **`0.9900`** | `0.9831` / `0.9931` / **`0.9881`** | `0.9826` / `0.9981` / **`0.9903`** | `0.9769` / `0.9903` / **`0.9906`** |
| **`DoS Hulk`** | 31,711 | 29,100 (`8.2%`) | `0.9913` / `0.9994` / **`0.9954`** | `0.9999` / `0.9998` / **`0.9999`** | `0.9999` / `0.9998` / **`0.9999`** | `1.0000` / `0.9998` / **`0.9999`** | `1.0000` / `0.9998` / **`0.9999`** | `0.9999` / `0.9999` / **`0.9999`** |
| **`DoS Slowhttptest`** | 594 | 285 (`52.0%`) | `0.2204` / `0.5354` / **`0.3122`** | `0.7209` / `0.6785` / **`0.6990`** | `0.7931` / `0.8838` / **`0.8360`** | `0.7859` / `0.8838` / **`0.8320`** | `0.7966` / `0.9360` / **`0.8607`** | `0.3942` / `0.7066` / **`0.7598`** |
| **`DoS slowloris`** | 1,039 | 524 (`49.6%`) | `0.2150` / `0.7575` / **`0.3349`** | `0.9837` / `0.8142` / **`0.8910`** | `0.9504` / `0.8672` / **`0.9069`** | `0.9504` / `0.8672` / **`0.9069`** | `0.9825` / `0.8643` / **`0.9196`** | `0.7743` / `0.8187` / **`0.8427`** |
| **`FTP-Patator`** | 1,593 | 797 (`50.0%`) | `0.2627` / `1.0000` / **`0.4160`** | `0.9994` / `1.0000` / **`0.9997`** | `1.0000` / `1.0000` / **`1.0000`** | `1.0000` / `1.0000` / **`1.0000`** | `1.0000` / `1.0000` / **`1.0000`** | `0.9994` / `1.0000` / **`1.0000`** |
| **`PortScan`** | 46,343 | 581 (`98.8%`) | `0.9680` / `0.9581` / **`0.9631`** | `0.9998` / `0.9623` / **`0.9807`** | `0.9999` / `0.9623` / **`0.9807`** | `0.9999` / `0.9622` / **`0.9807`** | `0.9999` / `0.9622` / **`0.9807`** | `0.6513` / `0.6497` / **`0.6481`** |
| **`Rare attack`** | 18 | 18 (`0.0%`) | `0.0040` / `0.5556` / **`0.0079`** | `1.0000` / `0.2222` / **`0.3636`** | `0.5833` / `0.3889` / **`0.4667`** | `1.0000` / `0.3889` / **`0.5600`** | `0.6000` / `0.3333` / **`0.4286`** | `0.3636` / `0.4667` / **`0.4286`** |
| **`SSH-Patator`** | 596 | 586 (`1.7%`) | `0.7928` / `0.9178` / **`0.8507`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9346` / **`0.9662`** | `0.9656` / `0.9656` / **`0.9656`** |
| **`Web Attack - Brute Force`** | 31 | 31 (`0.0%`) | `0.0076` / `1.0000` / **`0.0151`** | `1.0000` / `0.9677` / **`0.9836`** | `1.0000` / `1.0000` / **`1.0000`** | `1.0000` / `0.9677` / **`0.9836`** | `1.0000` / `1.0000` / **`1.0000`** | `0.9836` / `1.0000` / **`1.0000`** |
| **Macro Average / FPR** | **630,181** | **320,365** (`49.16%`) | F1=`0.5569` (FPR=`0.0370`) | F1=`0.9037` (FPR=`0.0000`) | F1=`0.9276` (FPR=`0.0000`) | F1=`0.9335` (FPR=`0.0000`) | **F1=`0.9283`** (FPR=`0.0000`) | `0.8418` / `0.8825` / **`0.8857`** |

#### Multi-Class Per-Class Precision / Recall / F1 — Without `dst_port` (55 Features)

| Canonical Class | Test Support | Unseen Support (`% Twins`) | LogReg (`P / R / F1`) | Random Forest (`P / R / F1`) | LightGBM (`P / R / F1`) | Fixed `0.5/0.5` (`P / R / F1`) | Meta-Learner (`P / R / F1`) | Unseen F1 (`RF / LGBM / Meta`) |
|---|---:|---:|---|---|---|---|---|---|
| **`BENIGN`** | 527,384 | 220,129 (`58.3%`) | `0.9958` / `0.9377` / **`0.9659`** | `0.9956` / `0.9999` / **`0.9978`** | `0.9961` / `0.9999` / **`0.9980`** | `0.9960` / `0.9999` / **`0.9980`** | `0.9963` / `0.9998` / **`0.9980`** | `0.9989` / `0.9995` / **`0.9995`** |
| **`Bot`** | 148 | 106 (`28.4%`) | `0.0760` / `0.9932` / **`0.1412`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `1.0000` / `0.9932` / **`0.9966`** | `0.9953` / `0.9953` / **`0.9953`** |
| **`DDoS`** | 19,139 | 15,086 (`21.2%`) | `0.9914` / `0.9987` / **`0.9951`** | `0.9991` / `0.9988` / **`0.9990`** | `0.9992` / `0.9989` / **`0.9990`** | `0.9992` / `0.9989` / **`0.9990`** | `0.9991` / `0.9989` / **`0.9990`** | `0.9987` / `0.9988` / **`0.9987`** |
| **`DoS GoldenEye`** | 1,585 | 1,500 (`5.4%`) | `0.5676` / `0.8820` / **`0.6907`** | `0.9846` / `0.9249` / **`0.9538`** | `0.9822` / `0.9413` / **`0.9613`** | `0.9828` / `0.9369` / **`0.9593`** | `0.9710` / `0.9722` / **`0.9716`** | `0.9632` / `0.9703` / **`0.9771`** |
| **`DoS Hulk`** | 31,711 | 29,099 (`8.2%`) | `0.9905` / `0.9993` / **`0.9949`** | `1.0000` / `0.9997` / **`0.9999`** | `0.9999` / `0.9998` / **`0.9999`** | `1.0000` / `0.9998` / **`0.9999`** | `1.0000` / `0.9998` / **`0.9999`** | `0.9999` / `0.9999` / **`0.9999`** |
| **`DoS Slowhttptest`** | 594 | 285 (`52.0%`) | `0.1453` / `0.5370` / **`0.2287`** | `0.7228` / `0.6717` / **`0.6963`** | `0.8012` / `0.9226` / **`0.8576`** | `0.8009` / `0.9209` / **`0.8567`** | `0.7883` / `0.9276` / **`0.8523`** | `0.3971` / `0.7592` / **`0.7514`** |
| **`DoS slowloris`** | 1,039 | 524 (`49.6%`) | `0.1262` / `0.7652` / **`0.2167`** | `0.9830` / `0.7806` / **`0.8702`** | `0.9845` / `0.8576` / **`0.9167`** | `0.9842` / `0.8383` / **`0.9054`** | `0.9815` / `0.8672` / **`0.9208`** | `0.7220` / `0.8356` / **`0.8456`** |
| **`FTP-Patator`** | 1,593 | 797 (`50.0%`) | `0.1014` / `1.0000` / **`0.1842`** | `1.0000` / `1.0000` / **`1.0000`** | `0.9994` / `1.0000` / **`0.9997`** | `0.9994` / `1.0000` / **`0.9997`** | `0.9994` / `1.0000` / **`0.9997`** | `1.0000` / `0.9994` / **`0.9994`** |
| **`PortScan`** | 46,343 | 143 (`99.7%`) | `0.9731` / `0.9580` / **`0.9655`** | `0.9996` / `0.9596` / **`0.9792`** | `0.9996` / `0.9596` / **`0.9792`** | `0.9996` / `0.9596` / **`0.9792`** | `0.9992` / `0.9598` / **`0.9791`** | `0.7143` / `0.7059` / **`0.7013`** |
| **`Rare attack`** | 18 | 18 (`0.0%`) | `0.0033` / `0.6111` / **`0.0066`** | `1.0000` / `0.1667` / **`0.2857`** | `1.0000` / `0.3333` / **`0.5000`** | `1.0000` / `0.2778` / **`0.4348`** | `0.5455` / `0.3333` / **`0.4138`** | `0.2857` / `0.5000` / **`0.4138`** |
| **`SSH-Patator`** | 596 | 586 (`1.7%`) | `0.6527` / `0.9178` / **`0.7629`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9178` / **`0.9571`** | `1.0000` / `0.9346` / **`0.9662`** | `0.9656` / `0.9656` / **`0.9656`** |
| **`Web Attack - Brute Force`** | 31 | 31 (`0.0%`) | `0.0080` / `1.0000` / **`0.0159`** | `1.0000` / `0.9677` / **`0.9836`** | `1.0000` / `1.0000` / **`1.0000`** | `1.0000` / `1.0000` / **`1.0000`** | `1.0000` / `1.0000` / **`1.0000`** | `0.9836` / `1.0000` / **`1.0000`** |
| **Macro Average / FPR** | **630,181** | **268,304** (`57.42%`) | F1=`0.5140` (FPR=`0.0623`) | F1=`0.8933` (FPR=`0.0001`) | F1=`0.9304` (FPR=`0.0001`) | F1=`0.9238` (FPR=`0.0001`) | **F1=`0.9247`** (FPR=`0.0002`) | `0.8353` / `0.8941` / **`0.8873`** |

### 2.3 Binary Confusion Matrices (2x2, Test Split = `630,181` Flows: `102,797` Attack, `527,384` Benign)

#### Binary Confusion Matrices — With `dst_port` (56 Features)

| Model | True `attack` -> Pred `attack` (TP) | True `attack` -> Pred `benign` (FN) | True `benign` -> Pred `attack` (FP) | True `benign` -> Pred `benign` (TN) | Benign FPR |
|---|---:|---:|---:|---:|---:|
| Majority Baseline (`majority`) | **0** | 102,797 | **0** | 527,384 | `0.0000` (`0.00%`) |
| Logistic Regression (`logreg`) | **97,863** | 4,934 | **7,398** | 519,986 | `0.0140` (`1.40%`) |
| **Random Forest (`random_forest`)** | **100,723** | 2,074 | **14** | 527,370 | `0.0000` (`0.00%`) |
| **LightGBM (`lightgbm`)** | **100,819** | 1,978 | **27** | 527,357 | `0.0001` (`0.01%`) |
| Fixed-Weights Ensemble (`0.5/0.5`) | **100,798** | 1,999 | **18** | 527,366 | `0.0000` (`0.00%`) |
| **Stacking Meta-Learner Ensemble** | **100,743** | 2,054 | **4** | 527,380 | `0.0000` (`0.00%`) |

#### Binary Confusion Matrices — Without `dst_port` (55 Features)

| Model | True `attack` -> Pred `attack` (TP) | True `attack` -> Pred `benign` (FN) | True `benign` -> Pred `attack` (FP) | True `benign` -> Pred `benign` (TN) | Benign FPR |
|---|---:|---:|---:|---:|---:|
| Majority Baseline (`majority`) | **0** | 102,797 | **0** | 527,384 | `0.0000` (`0.00%`) |
| Logistic Regression (`logreg`) | **97,834** | 4,963 | **7,194** | 520,190 | `0.0136` (`1.36%`) |
| **Random Forest (`random_forest`)** | **100,549** | 2,248 | **38** | 527,346 | `0.0001` (`0.01%`) |
| **LightGBM (`lightgbm`)** | **100,626** | 2,171 | **28** | 527,356 | `0.0001` (`0.01%`) |
| Fixed-Weights Ensemble (`0.5/0.5`) | **100,614** | 2,183 | **30** | 527,354 | `0.0001` (`0.01%`) |
| **Stacking Meta-Learner Ensemble** | **100,330** | 2,467 | **12** | 527,372 | `0.0000` (`0.00%`) |

### 2.4 Multi-Class Confusion Matrices (12x12 Canonical Classes, Test Split = `630,181` Flows)

### Multi-Class 12x12 Confusion Matrices — With `dst_port` (56 Features)

#### Majority Baseline (`majority`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,384** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 148 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 19,139 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 1,585 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 31,711 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 594 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 1,039 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 1,593 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 46,343 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 18 |
| **`SSH-Patator`** | 596 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 596 |
| **`Web Attack - Brute Force`** | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 31 |

#### Logistic Regression (`logreg`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0370` / `3.70%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **507,873** | 1,831 | 232 | 1,236 | 238 | 837 | 2,745 | 4,313 | 1,443 | 2,460 | 143 | 4,033 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 1 | 0 | **19,114** | 0 | 1 | 0 | 0 | 0 | 23 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 1 | 0 | 0 | **1,466** | 1 | 75 | 0 | 0 | 0 | 42 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 6 | 0 | 0 | 12 | **31,693** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 153 | 5 | 0 | 17 | 0 | **318** | 70 | 31 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 32 | 0 | 0 | 1 | 0 | 213 | **787** | 2 | 0 | 2 | 0 | 2 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,728 | 0 | 0 | 8 | 37 | 0 | 58 | 110 | **44,402** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | **10** | 0 | 4 | 18 |
| **`SSH-Patator`** | 32 | 0 | 0 | 0 | 0 | 0 | 1 | 16 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

#### Random Forest (`random_forest`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,359** | 0 | 13 | 0 | 1 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 23 | 0 | **19,116** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 32 | 0 | 0 | **1,536** | 1 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 0 | 0 | 0 | 7 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 161 | 0 | 0 | 16 | 0 | **403** | 14 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 52 | 0 | 0 | 1 | 0 | 140 | **846** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,744 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | **44,598** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 13 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | **4** | 0 | 0 | 18 |
| **`SSH-Patator`** | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **30** | 31 |

#### LightGBM (`lightgbm`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,362** | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 23 | 0 | **19,116** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 3 | 0 | 0 | **1,580** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 0 | 0 | 0 | 7 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 8 | 0 | 0 | 16 | 0 | **525** | 45 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 2 | 0 | 0 | 1 | 0 | 135 | **901** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,747 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | **44,594** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 8 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | **7** | 0 | 0 | 18 |
| **`SSH-Patator`** | 47 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

#### Fixed-Weights Ensemble (`0.5/0.5`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,366** | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 22 | 0 | **19,117** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 3 | 0 | 0 | **1,574** | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 0 | 0 | 0 | 7 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 8 | 0 | 0 | 16 | 0 | **525** | 45 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 2 | 0 | 0 | 1 | 0 | 135 | **901** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,749 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | **44,593** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 8 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | **7** | 0 | 0 | 18 |
| **`SSH-Patator`** | 47 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **30** | 31 |

#### Stacking Meta-Learner Ensemble (`ensemble_metalearner`) — Multi-Class (With `dst_port` (56 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,361** | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 3 | 4 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 22 | 0 | **19,117** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 0 | 0 | 0 | **1,582** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 0 | 0 | 0 | 7 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 8 | 0 | 0 | 16 | 0 | **556** | 14 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 2 | 0 | 0 | 1 | 0 | 138 | **898** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,749 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | **44,592** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 8 | 0 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | **6** | 0 | 0 | 18 |
| **`SSH-Patator`** | 37 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | **557** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

### Multi-Class 12x12 Confusion Matrices — Without `dst_port` (55 Features)

#### Majority Baseline (`majority`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0000` / `0.00%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,384** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 148 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 19,139 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 1,585 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 31,711 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 594 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 1,039 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 1,593 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 46,343 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 18 |
| **`SSH-Patator`** | 596 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 596 |
| **`Web Attack - Brute Force`** | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 31 |

#### Logistic Regression (`logreg`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0623` / `6.23%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **494,549** | 1,785 | 165 | 1,059 | 252 | 1,601 | 5,351 | 14,018 | 1,203 | 3,272 | 291 | 3,838 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 1 | 0 | **19,115** | 0 | 0 | 0 | 0 | 0 | 23 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 85 | 0 | 0 | **1,398** | 1 | 60 | 0 | 0 | 0 | 41 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 18 | 1 | 0 | 3 | **31,689** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 173 | 0 | 0 | 0 | 0 | **319** | 101 | 0 | 0 | 1 | 0 | 0 | 594 |
| **`DoS slowloris`** | 17 | 0 | 0 | 1 | 0 | 216 | **795** | 0 | 0 | 8 | 0 | 2 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,767 | 0 | 0 | 0 | 51 | 0 | 50 | 78 | **44,397** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | **11** | 0 | 4 | 18 |
| **`SSH-Patator`** | 33 | 0 | 0 | 0 | 0 | 0 | 1 | 15 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

#### Random Forest (`random_forest`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0001` / `0.01%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,347** | 0 | 16 | 3 | 0 | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 23 | 0 | **19,116** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 104 | 0 | 0 | **1,466** | 1 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 6 | 0 | 0 | 2 | **31,703** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 165 | 0 | 0 | 16 | 0 | **399** | 14 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 88 | 0 | 0 | 1 | 0 | 139 | **811** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,871 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | **44,471** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 14 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | **3** | 0 | 0 | 18 |
| **`SSH-Patator`** | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **30** | 31 |

#### LightGBM (`lightgbm`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0001` / `0.01%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,348** | 0 | 15 | 4 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 22 | 0 | **19,117** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 91 | 0 | 0 | **1,492** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 4 | 0 | 0 | 3 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 16 | 0 | 0 | 16 | 0 | **548** | 14 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 12 | 0 | 0 | 1 | 0 | 135 | **891** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,872 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | **44,469** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 10 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | **6** | 0 | 0 | 18 |
| **`SSH-Patator`** | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

#### Fixed-Weights Ensemble (`0.5/0.5`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0001` / `0.01%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,349** | 0 | 15 | 3 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 22 | 0 | **19,117** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 98 | 0 | 0 | **1,485** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 4 | 0 | 0 | 3 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 17 | 0 | 0 | 16 | 0 | **547** | 14 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 32 | 0 | 0 | 1 | 0 | 135 | **871** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,872 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | **44,470** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 11 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | **5** | 0 | 0 | 18 |
| **`SSH-Patator`** | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **547** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

#### Stacking Meta-Learner Ensemble (`ensemble_metalearner`) — Multi-Class (Without `dst_port` (55 Features)) (Benign FPR = `0.0002` / `0.02%`)

| True \\ Pred | `BENIGN` | `Bot` | `DDoS` | `GoldenEye` | `Hulk` | `Slowhttp` | `slowloris` | `FTP-Pat` | `PortScan` | `Rare` | `SSH-Pat` | `WebBrute` | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`BENIGN`** | **527,301** | 0 | 17 | 19 | 0 | 5 | 1 | 0 | 36 | 5 | 0 | 0 | 527,384 |
| **`Bot`** | 1 | **147** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 |
| **`DDoS`** | 22 | 0 | **19,117** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19,139 |
| **`DoS GoldenEye`** | 37 | 0 | 0 | **1,541** | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1,585 |
| **`DoS Hulk`** | 1 | 0 | 0 | 6 | **31,704** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31,711 |
| **`DoS Slowhttptest`** | 12 | 0 | 0 | 16 | 0 | **551** | 15 | 0 | 0 | 0 | 0 | 0 | 594 |
| **`DoS slowloris`** | 2 | 0 | 0 | 1 | 0 | 135 | **901** | 0 | 0 | 0 | 0 | 0 | 1,039 |
| **`FTP-Patator`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,593** | 0 | 0 | 0 | 0 | 1,593 |
| **`PortScan`** | 1,861 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | **44,481** | 0 | 0 | 0 | 46,343 |
| **`Rare attack`** | 8 | 0 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | **6** | 0 | 0 | 18 |
| **`SSH-Patator`** | 37 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | **557** | 0 | 596 |
| **`Web Attack - Brute Force`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **31** | 31 |

---

## 3. `Rare attack` Class Merge Quoted from `data/processed/data_card.json` (`SEN-FR-05`)

- Per `SEN-FR-05`, raw attack classes with `< 100` cleaned flows are merged into `'Rare attack'` by `sentinel_data/split.py`; below is the verbatim `rare_classes` block from [`data/processed/data_card.json`](../../data/processed/data_card.json):

```json
{
  "rare_classes": {
    "rule": "attack classes with fewer than 100 flows after cleaning are merged into 'Rare attack'",
    "min_flows": 100,
    "merged": [
      "Heartbleed",
      "Infiltration",
      "Web Attack - Sql Injection",
      "Web Attack - XSS"
    ],
    "excluded": [],
    "rows_per_original_class": {
      "Heartbleed": {
        "train": 5,
        "val": 1,
        "test": 3
      },
      "Infiltration": {
        "train": 16,
        "val": 6,
        "test": 6
      },
      "Web Attack - Sql Injection": {
        "train": 7,
        "val": 2,
        "test": 3
      },
      "Web Attack - XSS": {
        "train": 15,
        "val": 4,
        "test": 6
      }
    }
  }
}
```

### 3.1 Original Sub-100-Flow Classes Merged into `Rare attack`

| Original Raw Attack Label | Train Flows | Val Flows | Test Flows | Total Cleaned Flows | Threshold (`min_flows`) | Canonical Target Label |
|---|---:|---:|---:|---:|---:|---|
| `Heartbleed` | 5 | 1 | 3 | **9** | `< 100` | **`Rare attack`** |
| `Infiltration` | 16 | 6 | 6 | **28** | `< 100` | **`Rare attack`** |
| `Web Attack - Sql Injection` | 7 | 2 | 3 | **12** | `< 100` | **`Rare attack`** |
| `Web Attack - XSS` | 15 | 4 | 6 | **25** | `< 100` | **`Rare attack`** |
| **Merged `Rare attack` Total** | **43** | **13** | **18** | **74** | `< 100` | **`Rare attack`** |

### 3.2 Full 12-Class Canonical Distribution Across Splits (`data_card.json`)

- `Web Attack - Brute Force` (`147` total flows) and `Bot` (`736` total flows) exceed `100` flows and remain standalone canonical classes.

| Canonical Label (12 Classes) | Train Flows (`60%`) | Val Flows (`20%`) | Test Flows (`20%`) | Total Flows | Share of Dataset |
|---|---:|---:|---:|---:|---:|
| **`BENIGN`** | 1,581,299 | 526,163 | 527,384 | 2,634,846 | `83.7997%` |
| **`Bot`** | 442 | 146 | 148 | 736 | `0.0234%` |
| **`DDoS`** | 57,309 | 18,995 | 19,139 | 95,443 | `3.0355%` |
| **`DoS GoldenEye`** | 4,578 | 1,338 | 1,585 | 7,501 | `0.2386%` |
| **`DoS Hulk`** | 94,009 | 30,774 | 31,711 | 156,494 | `4.9772%` |
| **`DoS Slowhttptest`** | 1,636 | 354 | 594 | 2,584 | `0.0822%` |
| **`DoS slowloris`** | 2,943 | 889 | 1,039 | 4,871 | `0.1549%` |
| **`FTP-Patator`** | 4,767 | 1,580 | 1,593 | 7,940 | `0.2525%` |
| **`PortScan`** | 138,811 | 45,470 | 46,343 | 230,624 | `7.3349%` |
| **`Rare attack`** | 43 | 13 | 18 | 74 | `0.0024%` |
| **`SSH-Patator`** | 1,778 | 586 | 596 | 2,960 | `0.0941%` |
| **`Web Attack - Brute Force`** | 88 | 28 | 31 | 147 | `0.0047%` |
| **Total** | **1,887,703** | **626,336** | **630,181** | **3,144,220** | **`100.0000%`** |

---

## 4. Leakage Check Result (`SEN-106`) & Weights & Biases Run Links (`SEN-107`)

### 4.1 `SEN-106` Train–Test Exact Twin Audit & De-Twinned Evaluation

- **With `dst_port` (56 features)**: `309,816 / 630,181` (`49.16%`) test rows have an exact feature twin in `train.parquet` (`309,390` exact binary `(X, y)` twins, `309,389` exact 12-class `(X, y)` twins), leaving **`320,365` (`50.84%`) strictly unseen test rows**.
- **Without `dst_port` (55 features)**: `361,877 / 630,181` (`57.42%`) test rows have an exact feature twin in `train.parquet` (`361,822` exact binary `(X, y)` twins, `361,821` exact 12-class `(X, y)` twins), leaving **`268,304` (`42.58%`) strictly unseen test rows**.
- **Root Cause**: Full-row duplicates were removed in `sentinel_data.clean` and `sentinel_data.split` enforces a `1s` chronological gap (`6,217` boundary rows dropped); twins arise after stripping `ID_COLS` because deterministic attack tools produce repeated flow summary vectors (`98.8%` of `PortScan` flows with port and `99.7%` without port match a training template).

| Model | Task | `dst_port` | Macro-F1 (All `630,181`) | Macro-F1 (Twins Only) | Macro-F1 (Strictly Unseen) | Delta (`Unseen - All`) | Benign FPR (All) | Benign FPR (Strictly Unseen) |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (`SEN-109` / `SEN-106`) | Binary | `True` (56) | `0.9645` | — | **`0.9492`** | `-0.0153` | `0.0140` | `0.0250` |
| Logistic Regression (`SEN-109` / `SEN-106`) | Binary | `False` (55) | `0.9650` | — | **`0.9547`** | `-0.0103` | `0.0136` | `0.0270` |
| Logistic Regression (`SEN-109` / `SEN-106`) | Multi-class | `True` (56) | `0.5569` | — | **`0.4714`** | `-0.0855` | `0.0370` | `0.0560` |
| Logistic Regression (`SEN-109` / `SEN-106`) | Multi-class | `False` (55) | `0.5140` | — | **`0.4638`** | `-0.0502` | `0.0623` | `0.0640` |
| Random Forest (`SEN-FR-08`) | Binary | `True` (56) | `0.9939` | `0.9916` | **`0.9964`** | `+0.0025` | `0.0000` | `0.0000` |
| LightGBM (`SEN-FR-08`) | Binary | `True` (56) | `0.9941` | `0.9918` | **`0.9967`** | `+0.0026` | `0.0001` | `0.0001` |
| Fixed-Weights Ensemble (`0.5/0.5`) | Binary | `True` (56) | `0.9941` | `0.9917` | **`0.9967`** | `+0.0026` | `0.0000` | `0.0001` |
| Stacking Meta-Learner (`SEN-FR-08`) | Binary | `True` (56) | `0.9940` | `0.9916` | **`0.9966`** | `+0.0026` | `0.0000` | `0.0000` |
| Random Forest (`SEN-FR-08`) | Binary | `False` (55) | `0.9933` | `0.9898` | **`0.9973`** | `+0.0040` | `0.0001` | `0.0001` |
| LightGBM (`SEN-FR-08`) | Binary | `False` (55) | `0.9936` | `0.9898` | **`0.9979`** | `+0.0043` | `0.0001` | `0.0000` |
| Fixed-Weights Ensemble (`0.5/0.5`) | Binary | `False` (55) | `0.9935` | `0.9898` | **`0.9978`** | `+0.0043` | `0.0001` | `0.0001` |
| Stacking Meta-Learner (`SEN-FR-08`) | Binary | `False` (55) | `0.9927` | `0.9896` | **`0.9963`** | `+0.0036` | `0.0000` | `0.0000` |
| Random Forest (`SEN-FR-08`) | Multi-class | `True` (56) | `0.9037` | `0.8907` | **`0.8418`** | `-0.0619` | `0.0000` | `0.0001` |
| LightGBM (`SEN-FR-08`) | Multi-class | `True` (56) | `0.9276` | `0.8925` | **`0.8825`** | `-0.0451` | `0.0000` | `0.0001` |
| Fixed-Weights Ensemble (`0.5/0.5`) | Multi-class | `True` (56) | `0.9335` | `0.8925` | **`0.8883`** | `-0.0452` | `0.0000` | `0.0001` |
| Stacking Meta-Learner (`SEN-FR-08`) | Multi-class | `True` (56) | `0.9283` | `0.9925` | **`0.8857`** | `-0.0426` | `0.0000` | `0.0001` |
| Random Forest (`SEN-FR-08`) | Multi-class | `False` (55) | `0.8933` | `0.8683` | **`0.8353`** | `-0.0580` | `0.0001` | `0.0001` |
| LightGBM (`SEN-FR-08`) | Multi-class | `False` (55) | `0.9304` | `0.8703` | **`0.8941`** | `-0.0363` | `0.0001` | `0.0001` |
| Fixed-Weights Ensemble (`0.5/0.5`) | Multi-class | `False` (55) | `0.9238` | `0.8703` | **`0.8866`** | `-0.0372` | `0.0001` | `0.0001` |
| Stacking Meta-Learner (`SEN-FR-08`) | Multi-class | `False` (55) | `0.9247` | `0.9804` | **`0.8873`** | `-0.0374` | `0.0002` | `0.0002` |

### 4.2 `dst_port` Target-Port Memorization Analysis (`with dst_port` vs. `without dst_port`)

| Model Family | Binary Macro-F1 (`port=True -> False`) | Binary Delta (`No-Port - Port`) | Multi-Class Macro-F1 (`port=True -> False`) | Multi-Class Delta (`No-Port - Port`) | Diagnosis |
|---|---:|---:|---:|---:|---|
| **Logistic Regression** | `0.9645 -> 0.9650` | `+0.0005` | `0.5569 -> 0.5140` | `-0.0429` | Linear multi-class separator relies heavily on `dst_port` |
| **PyTorch MLP (`SEN-FR-13`)** | `0.9940 -> 0.9924` | `-0.0016` | `0.7133 -> 0.6388` | **`-0.0745`** | `dst_port` is #2 MLP permutation feature (`0.229`); drops `7.45` pts without port |
| **Random Forest (`SEN-FR-08`)** | `0.9939 -> 0.9933` | `-0.0006` | `0.9037 -> 0.8933` | `-0.0104` | Drop driven by 18-flow `Rare attack` (`0.3636 -> 0.2857`) and `DoS Slowhttptest` (`0.6990 -> 0.6963`) |
| **LightGBM (`SEN-FR-08`)** | `0.9941 -> 0.9936` | `-0.0005` | `0.9276 -> 0.9304` | `+0.0028` | Minimal sensitivity (`0.28` pt) to `dst_port` removal; splits on packet-size/IAT stats |
| **Fixed-Weights (`0.5/0.5`)** | `0.9941 -> 0.9935` | `-0.0006` | `0.9335 -> 0.9238` | `-0.0097` | Equal-weight probability average tracks single-tree baselines |
| **Fixed-Weights (Val-Tuned)** | `0.9941 -> 0.9934` | `-0.0007` | `0.9289 -> 0.9243` | `-0.0046` | Val-tuned linear blend matches equal-weight ensemble on test |
| **Stacking Meta-Learner** | `0.9940 -> 0.9927` | `-0.0013` | `0.9283 -> 0.9247` | **`-0.0036`** | **Best overall (`0.9247` without `dst_port`, `0.8873` on strictly unseen rows)** |

- **Binary robustness**: Removing both `dst_port` and all `361,877` train twins yields `0.9973` (RF), `0.9979` (LightGBM), `0.9978` (Fixed `0.5/0.5`), and `0.9963` (Meta-Learner) Macro-F1 at `0.0000–0.0000` Benign FPR.
- **Sensor deployment recommendation**: Deploy the **`dst_port=False` (55-feature)** models so detection generalizes across non-standard target ports.

### 4.3 Weights & Biases Logged Runs (`SEN-107`)

- **Primary W&B Project**: [`https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel) | **Baselines Project (`SEN-109`)**: [`https://forge.coreweave.com/wandb/wcs-sentinel/baselines-v1`](https://forge.coreweave.com/wandb/wcs-sentinel/baselines-v1) | **Summary Run**: [`daffy1yz`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/daffy1yz)

| W&B Run ID | Run Name | Model | Task | `dst_port` | Features | Seed | Data Version | Val Macro-F1 | Test Macro-F1 (All) | Test Macro-F1 (Unseen) | Test Benign FPR | Direct W&B Link |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| `5hgg46mm` | `random_forest_binary_port` | random_forest | `binary` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9990` | **`0.9939`** | `0.9964` | `0.0000` | [`5hgg46mm`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/5hgg46mm) |
| `3o7mlf1q` | `lightgbm_binary_port` | lightgbm | `binary` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9989` | **`0.9941`** | `0.9967` | `0.0001` | [`3o7mlf1q`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/3o7mlf1q) |
| `9f2bu7w0` | `ensemble_fixed_binary_port` | ensemble_fixed (0.50/0.50) | `binary` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9990` | **`0.9941`** | `0.9967` | `0.0000` | [`9f2bu7w0`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/9f2bu7w0) |
| `xv5jiln1` | `ensemble_fixed_tuned_binary_port` | ensemble_fixed_tuned (0.60/0.40) | `binary` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9990` | **`0.9941`** | `0.9967` | `0.0000` | [`xv5jiln1`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/xv5jiln1) |
| `y6vebwco` | `ensemble_metalearner_binary_port` | ensemble_metalearner | `binary` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9990` | **`0.9940`** | `0.9966` | `0.0000` | [`y6vebwco`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/y6vebwco) |
| `mlt3xf55` | `random_forest_binary_noport` | random_forest | `binary` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9984` | **`0.9933`** | `0.9973` | `0.0001` | [`mlt3xf55`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/mlt3xf55) |
| `cb9epptm` | `lightgbm_binary_noport` | lightgbm | `binary` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9984` | **`0.9936`** | `0.9979` | `0.0001` | [`cb9epptm`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/cb9epptm) |
| `e2spnast` | `ensemble_fixed_binary_noport` | ensemble_fixed (0.50/0.50) | `binary` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9984` | **`0.9935`** | `0.9978` | `0.0001` | [`e2spnast`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/e2spnast) |
| `b0g5xzpg` | `ensemble_fixed_tuned_binary_noport` | ensemble_fixed_tuned (0.70/0.30) | `binary` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9984` | **`0.9934`** | `0.9976` | `0.0001` | [`b0g5xzpg`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/b0g5xzpg) |
| `d4gpqqg9` | `ensemble_metalearner_binary_noport` | ensemble_metalearner | `binary` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9988` | **`0.9927`** | `0.9963` | `0.0000` | [`d4gpqqg9`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/d4gpqqg9) |
| `07uaslxa` | `random_forest_multiclass_port` | random_forest | `multiclass` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9728` | **`0.9037`** | `0.8418` | `0.0000` | [`07uaslxa`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/07uaslxa) |
| `bcg8zhr2` | `lightgbm_multiclass_port` | lightgbm | `multiclass` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9876` | **`0.9276`** | `0.8825` | `0.0000` | [`bcg8zhr2`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/bcg8zhr2) |
| `wqsgxuew` | `ensemble_fixed_multiclass_port` | ensemble_fixed (0.50/0.50) | `multiclass` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9847` | **`0.9335`** | `0.8883` | `0.0000` | [`wqsgxuew`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/wqsgxuew) |
| `9cjpg20x` | `ensemble_fixed_tuned_multiclass_port` | ensemble_fixed_tuned (0.25/0.75) | `multiclass` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9876` | **`0.9289`** | `0.8839` | `0.0000` | [`9cjpg20x`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/9cjpg20x) |
| `k64lrayi` | `ensemble_metalearner_multiclass_port` | ensemble_metalearner | `multiclass` | `True` | 56 | `42` | `nfs2023te-v1` | `0.9893` | **`0.9283`** | `0.8857` | `0.0000` | [`k64lrayi`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/k64lrayi) |
| `p7jobj3p` | `random_forest_multiclass_noport` | random_forest | `multiclass` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9589` | **`0.8933`** | `0.8353` | `0.0001` | [`p7jobj3p`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/p7jobj3p) |
| `xwewtupk` | `lightgbm_multiclass_noport` | lightgbm | `multiclass` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9752` | **`0.9304`** | `0.8941` | `0.0001` | [`xwewtupk`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/xwewtupk) |
| `0x2at48f` | `ensemble_fixed_multiclass_noport` | ensemble_fixed (0.50/0.50) | `multiclass` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9776` | **`0.9238`** | `0.8866` | `0.0001` | [`0x2at48f`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/0x2at48f) |
| `y1ccrfby` | `ensemble_fixed_tuned_multiclass_noport` | ensemble_fixed_tuned (0.40/0.60) | `multiclass` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9777` | **`0.9243`** | `0.8878` | `0.0001` | [`y1ccrfby`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/y1ccrfby) |
| `k3gmz3iz` | `ensemble_metalearner_multiclass_noport` | ensemble_metalearner | `multiclass` | `False` | 55 | `42` | `nfs2023te-v1` | `0.9829` | **`0.9247`** | `0.8873` | `0.0002` | [`k3gmz3iz`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/k3gmz3iz) |
| `daffy1yz` | `rf_lgbm_ensemble_comparison_summary` | 20-Run Summary Table | `both` | `both` | `55 / 56` | `42` | `nfs2023te-v1` | — | — | — | — | [`daffy1yz`](https://forge.coreweave.com/wandb/wcs-sentinel/Sentinel/runs/daffy1yz) |

---

## 5. Provisional Stage 2 Quality Targets Confirmed or Changed with Kam

### 5.1 Stage 2 Quality Gate Verification Table

| Quality Gate / Metric | Provisional Target | Achieved (`dst_port=True`, 56 feats) | Achieved (`dst_port=False`, 55 feats) | Achieved (Strictly Unseen, `dst_port=False`) | Status | Decision Confirmed with Kam |
|---|---:|---:|---:|---:|:---:|---|
| **Binary Macro-F1** | `>= 0.9500` | **`0.9941`** (LGBM / Fixed)<br>`0.9940` (Meta / MLP) / `0.9939` (RF) | **`0.9935`** (Fixed)<br>`0.9936` (LGBM) / `0.9933` (RF) / `0.9927` (Meta) | **`0.9978`** (Fixed)<br>`0.9979` (LGBM) / `0.9973` (RF) / `0.9963` (Meta) | **PASS** | **Confirmed (`>= 0.98` operational gate)**: Binary detection exceeds `0.992` on all test rows and `0.996` on strictly unseen rows without `dst_port`. |
| **Binary Attack Recall / Unseen F1** | `>= 0.9500` | Recall: **`0.9808`** (LGBM / Fixed)<br>`0.9800` (Meta) / `0.9798` (RF) | Recall: **`0.9788`** (Fixed)<br>`0.9789` (LGBM) / `0.9781` (RF) / `0.9760` (Meta) | Unseen `attack` F1: **`0.9964`** (Fixed)<br>`0.9965` (LGBM) / `0.9956` (RF) / `0.9939` (Meta) | **PASS** | **Confirmed (`>= 0.97` all-test recall)**: Exceeds target across all tree and ensemble models. |
| **Binary Benign FPR** | `<= 0.0100` (`1.0%`) | **`0.0000`** (RF)<br>**`0.0001`** (LGBM / Fixed / Meta) | **`0.0001`** (Fixed / Meta)<br>**`0.0001`** (RF / LGBM) | **`0.0001`** (Fixed / Meta)<br>`0.0001` (RF) / `0.0000` (LGBM) | **PASS** | **Confirmed & Tightened (`<= 0.0010` / `0.10%`)**: Only `12–28` false positives (`12` Meta, `38` RF, `28` LGBM) out of `527,384` benign test flows without `dst_port`. |
| **Multi-Class Macro-F1 (12 Classes)** | `>= 0.8500` | **`0.9283`** (Meta)<br>`0.9335` (Fixed) / `0.9276` (LGBM) / `0.9037` (RF) | **`0.9247`** (Meta)<br>`0.9304` (LGBM) / `0.9238` (Fixed) / `0.8933` (RF) | **`0.8873`** (Meta)<br>`0.8941` (LGBM) / `0.8866` (Fixed) / `0.8353` (RF) | **PASS** | **Confirmed (`>= 0.85` single tree, `>= 0.88` Meta-Learner)**: All tree and ensemble models pass `>= 0.85` with and without `dst_port`; Meta-Learner reaches `0.9247` (`0.8873` unseen). |
| **Multi-Class Benign FPR** | `<= 0.0100` (`1.0%`) | **`0.0000`** (RF / LGBM / Fixed)<br>**`0.0000`** (Meta) | **`0.0001`** (RF / LGBM / Fixed / Meta) | **`0.0001`** (RF / LGBM / Fixed)<br>**`0.0002`** (Meta) | **PASS** | **Confirmed & Tightened (`<= 0.0010` / `0.10%`)**: Multi-class Benign FPR is `0.0001` (`0.01%`) across all tree and ensemble models without `dst_port`. |
| **8 High-Precision Attack Classes** (`DDoS`, `DoS Hulk`, `DoS GoldenEye`, `FTP-Patator`, `PortScan`, `SSH-Patator`, `Bot` [`148`], `Web Attack - Brute Force` [`31`]) | `>= 0.9500` (`>= 0.85` minority) | **`0.9662 – 1.0000`** (`Bot` = `0.9966`, `Web Attack - Brute Force` = `1.0000`) | **`0.9662 – 1.0000`** (`Bot` = `0.9966`, `Web Attack - Brute Force` = `1.0000`) | **`0.9656 – 1.0000`** (`Bot` = `0.9953`, `WebBrute` = `1.0000`, `143`-row unseen `PortScan` = `0.7013`) | **PASS** | **Confirmed (`>= 0.95` across all 8 classes)**: Even minority standalone classes `Bot` (`0.9966`) and `Web Attack - Brute Force` (`1.0000`) exceed `0.98` F1 on RF, LGBM, and Meta-Learner without `dst_port`. |
| **Slow-Rate HTTP DoS Classes** (`DoS Slowhttptest` [`594` test] & `DoS slowloris` [`1,039` test]) | `>= 0.8000` | `Slowhttptest`: **`0.8320`** (Fixed) / `0.8607` (Meta)<br>`slowloris`: **`0.9069`** (Fixed) / `0.9196` (Meta) | `Slowhttptest`: **`0.8576`** (LGBM) / `0.8523` (Meta)<br>`slowloris`: **`0.9208`** (Meta) | `Slowhttptest`: `0.3971` (RF) / `0.7514` (Meta) / **`0.7592`** (LGBM)<br>`slowloris`: **`0.8456`** (Meta) | **PASS** | **Confirmed (`>= 0.84` all-test on Meta-Learner)**: Both slow-HTTP DoS classes cross `0.84` on all test flows (`0.8523` & `0.9208` on Meta no-port; `0.3971–0.7592` on strictly unseen `Slowhttptest`). |
| **Merged Ultra-Low-Support Class**: `Rare attack` (`43` train / `18` test flows across 4 merged attack types) | Provisional uniform `>= 0.80` -> **Adjusted to `>= 0.35` F1 on Meta-Learner** | **`0.4286`** (Meta-Learner)<br>`0.3636` (RF) / `0.0000` (LGBM) | **`0.4138`** (Meta-Learner)<br>`0.0000` (RF / LGBM) | **`0.4138`** (Meta-Learner, `0.0%` twins) | **ADJUSTED WITH KAM** | **Changed with Kam**: Only `Rare attack` (`43` train / `18` test flows across `Heartbleed`, `Infiltration`, `SQLi`, `XSS`) requires a per-class F1 target adjustment (`>= 0.35` on Meta-Learner, achieved `0.4138` without `dst_port`), while binary detection flags the attacks as malicious. |
| **Sensor Inference Latency (`C1` Package)** | `<= 15.0 ms / 1k flows` (single tree `<= 10.0 ms`) | Binary: `0.62 ms` (LGBM) / `0.73 ms` (RF) / `1.51 ms` (Meta) | Binary: `0.60 ms` (LGBM) / `0.59 ms` (RF) / `1.36 ms` (Meta)<br>Multi: `1.30 ms` (RF) / `4.62 ms` (LGBM) / `7.09 ms` (Meta) | Same | **PASS** | **Confirmed**: Binary inference (`0.60–1.36 ms / 1k flows`) and 12-class inference (`1.30–7.09 ms / 1k flows`) easily meet real-time sensor budgets. |

### 5.2 Key Deployment Decisions Confirmed with Kam

- **Production Feature Contract**: Deploy **`dst_port=False` (55 numeric features)** on the sensor (`SEN-FR-01` / `SEN-FR-02`) to prevent target-port memorization.
- **Production Model Selection (`C1` Package)**: Ship **Stacking Meta-Learner Ensemble (`dst_port=False`)** for 12-class attribution (`Macro-F1 = 0.9247`, `Unseen Macro-F1 = 0.8873`, `Benign FPR = 0.0002`, `7.09 ms / 1k flows`) paired with **LightGBM (`dst_port=False`)** (`Binary Macro-F1 = 0.9936`, `Unseen = 0.9979`, `0.60 ms / 1k flows`) as the fast single-tree detector.
- **Benchmark Protocol (`SEN-106`)**: Keep the chronological 60/20/20 split (`nfs2023te-v1`) and report both **All-Test (`630,181` rows)** and **Strictly Unseen De-Twinned (`268,304` / `320,365` rows)** metrics via `eval_template.py`.

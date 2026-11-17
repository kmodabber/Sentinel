# SEN-106: Train–Test Twins Check

## Methodology & Twin Definition

To reproduce these numbers from the repository root (after `python -m sentinel_data.split` has produced `data/processed/{train,test}.parquet`):

```bash
python scripts/check_train_test_twins.py
```

- **Which columns count as a twin**:
  - Both `train.parquet` (`1,887,703` rows) and `test.parquet` (`630,181` rows) are passed through [`prep_xy()`](../../sentinel_ml/eval_template.py), which drops all 17 identifier/timestamp columns in `ID_COLS` (`id`, `expiration_id`, `flow_id`, `src_ip`, `dst_ip`, `src_mac`, `dst_mac`, `src_oui`, `dst_oui`, `bidirectional_first_seen_ms`, `bidirectional_last_seen_ms`, `src2dst_first_seen_ms`, `src2dst_last_seen_ms`, `dst2src_first_seen_ms`, `dst2src_last_seen_ms`, `timestamp`, `label`) as well as non-numeric split metadata (`day`, `label_original`).
  - **With `dst_port` (`use_port=True`)**: a test row is counted as an exact twin in `train` if all **56 numeric flow features** (including `dst_port`) **plus its target label** exactly match at least one row in `train.parquet`.
  - **Without `dst_port` (`use_port=False`)**: `dst_port` is additionally dropped, leaving **55 numeric flow features** **plus the target label**.
- **Model configuration**:
  - Matches `sentinel_ml/baselines.py` (`make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs"))`) trained on the full `train.parquet` split over the 12 canonical classes (`sentinel_features.features.LABELS`).

---

## Results

- **With `dst_port` (56 features)**:
  - **Exact twins in train**: `309,400 / 630,181` (**49.1%**) on `(56 features, binary label)` — leaves `320,781` strictly unseen test rows (`309,399 / 630,181` on `(56 features, 12-class label)` — leaves `320,782` strictly unseen test rows).
  - **Binary LogReg Macro-F1**: `0.9643` (all test; `0.9645` in `results/baselines_test.json`) -> **`0.9492`** (without twins, `-0.0151`). Attack recall goes from `95.2%` -> `95.8%`, benign FPR from `1.4%` -> `2.5%`.
  - **Multiclass LogReg Macro-F1 (12 canonical classes, `baselines.py` config)**: `0.5562` (all test; `0.5569` in `results/baselines_test.json`) -> **`0.4714`** (without twins, `-0.0848`). Benign FPR goes from `3.8%` -> `5.6%`. *(Note: the earlier `0.6245` -> `0.5787` figure came from an unweighted / subsampled exploratory run rather than the full-train `class_weight="balanced"` baseline setup.)*

- **Without `dst_port` (55 features)**:
  - **Exact twins in train**: `361,832 / 630,181` (**57.4%**) on `(55 features, binary label)` — leaves `268,349` strictly unseen test rows (`361,831 / 630,181` on `(55 features, 12-class label)` — leaves `268,350` strictly unseen test rows).
  - **Binary LogReg Macro-F1**: `0.9653` (all test; `0.9650` in `results/baselines_test.json`) -> **`0.9547`** (without twins, `-0.0106`). Attack recall goes from `95.2%` -> `97.0%`, benign FPR from `1.4%` -> `2.7%`.
  - **Multiclass LogReg Macro-F1 (12 canonical classes, `baselines.py` config)**: `0.5184` (all test; `0.5140` in `results/baselines_test.json`) -> **`0.4638`** (without twins, `-0.0546`). Benign FPR goes from `6.5%` -> `6.4%`. *(Earlier unweighted / subsampled exploratory run: `0.5858` -> `0.5628`.)*

---

## Takeaway (Follow-up on SEN-109 Leakage Warning)

1. **Why 49–57% of test rows have feature twins in train**:
   - Full-row duplicates were already removed during cleaning (`sentinel_data.clean`), and `sentinel_data.split` enforces a chronological 60/20/20 split with a 1-second gap so no single connection spans train and test.
   - Once host/session identifiers (`src_ip`, `dst_ip`, `src_port`, timestamps) are stripped by `ID_COLS`, many short or tool-generated network flows (e.g., single-packet PortScan probes, SYN/DNS queries, and fixed-rate DoS floods) produce identical 55/56-feature summary vectors across different points in time. These are **naturally repeated flow signatures**, not split contamination.
2. **Did train–test twins cause the SEN-109 binary LogReg warning (`macro_f1 >= 0.95`)?**:
   - **No.** Dropping every test row that has an exact feature+label twin in `train` only reduces Binary LogReg Macro-F1 by `0.0151` (with `dst_port`, to `0.9492`) and `0.0106` (without `dst_port`, to `0.9547`), while attack recall slightly *increases* (`95.2%` -> `95.8%` / `97.0%`). Binary benign vs. attack traffic in NFS-2023-TE is genuinely linearly separable on packet-size and inter-arrival statistics even on strictly unseen feature vectors.
   - Multiclass Macro-F1 drops more noticeably (`-0.0848` with `dst_port`, `-0.0546` without `dst_port`) because a large fraction of high-volume, deterministic attack classes (`PortScan`, `DoS Hulk`, `DDoS`) consists of repeated template flows; removing them leaves a higher proportion of harder, variable-length flows and rare classes.
3. **Reporting recommendation going forward**:
   - Keep the standard **full `test.parquet` split** as the primary benchmark across models (so class priors and traffic volume reflect realistic capture distributions and stay aligned with `eval_template.py` and `baselines_test.json`), and treat the de-twinned ("without twins") slice via `scripts/check_train_test_twins.py` as a supplementary diagnostic check.

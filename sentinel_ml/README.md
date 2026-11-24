# sentinel_ml

Machine learning models, baselines, evaluation templates, and ensemble comparisons for Sentinel network intrusion detection on the cleaned NFS-2023-TE flow dataset.

## Modules

- `baselines.py` (SEN-FR-07): Majority-class (`DummyClassifier`) and `LogisticRegression` sanity baselines with data leakage check across binary and multi-class tasks (`with` and `without` `dst_port`).
- `eval_template.py`: Universal evaluation harness (`run_eval_template`, `evaluate_single`, `find_train_twins`, `get_top_features`, `prep_xy`) computing Macro-F1, Benign FPR, per-class precision/recall/F1, confusion matrices, top feature importances, and `train_twin_analysis` (how many test rows have an exact feature twin in `train.parquet` and how Macro-F1 / Benign FPR / per-class F1 change on the strictly unseen test rows).
- `ensemble.py`:
  - `default_rf(task)` / `default_lgbm(task)`: Configured `RandomForestClassifier` and stabilized `LGBMClassifier` (with `max_delta_step`, `reg_lambda`, and `min_child_weight` to prevent Newton-step softmax overflow on rare attack classes).
  - `FixedWeightEnsemble`: Soft-voting ensemble combining Random Forest and LightGBM probabilities with fixed weights $w_{\text{RF}} P_{\text{RF}} + w_{\text{LGBM}} P_{\text{LGBM}}$ (plus `sweep_fixed_weights` for validation grid search).
  - `MetaLearnerEnsemble`: Stacking ensemble combining Random Forest and LightGBM probabilities, scaled log-probabilities, and model disagreement features via a `LogisticRegression` meta-learner and per-class Macro-F1 multiplier calibration trained on out-of-sample predictions.
- `train_and_compare.py`: End-to-end training, evaluation, and head-to-head comparison of Random Forest vs. LightGBM and Meta-Learner Ensemble vs. Fixed-Weights Ensemble across binary and multi-class detection, with Weights & Biases (`wcs-sentinel/Sentinel`) logging.
- `make_synthetic.py`: Synthetic flow Parquet generator for fast local unit testing.

## Usage

After running `sentinel_data.get_data`, `sentinel_data.clean`, and `sentinel_data.split` from the repo root:

```bash
# Train RF, LightGBM, Fixed-Weights Ensemble, and Meta-Learner Ensemble with W&B logging
python -m sentinel_ml.train_and_compare

# Run unit tests
pytest
```

## Outputs (`results/`)

- `eval_random_forest_binary.json` & `eval_random_forest_multiclass.json`
- `eval_lightgbm_binary.json` & `eval_lightgbm_multiclass.json`
- `eval_ensemble_fixed_binary.json` & `eval_ensemble_fixed_multiclass.json`
- `eval_ensemble_metalearner_binary.json` & `eval_ensemble_metalearner_multiclass.json`
- `rf_vs_lightgbm_comparison.json`
- `ensemble_comparison.json`

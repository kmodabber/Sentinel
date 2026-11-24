"""Ensemble estimators combining Random Forest and LightGBM: Fixed Weights vs. Meta-Learner (Stacking)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def _normalize_importances(scores: np.ndarray) -> np.ndarray:
    """Normalizes non-negative feature importance scores to sum to 1.0."""
    arr = np.asarray(scores, dtype=float)
    total = float(np.sum(np.abs(arr)))
    return np.abs(arr) / total if total > 0 else np.zeros_like(arr)


def default_rf(task: str = "multiclass", random_state: int = 42, n_estimators: int = 80) -> RandomForestClassifier:
    """Returns a well-regularized Random Forest configured for binary or multi-class flow detection."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        max_samples=0.25,
        class_weight=None,
        n_jobs=-1,
        random_state=random_state,
    )


def default_lgbm(task: str = "multiclass", random_state: int = 42, n_estimators: int = 100) -> LGBMClassifier:
    """Returns a stabilized LightGBM model configured for binary or multi-class flow detection.

    For multi-class detection with extreme rare classes (e.g. 43 Rare attack rows out of 1.89M),
    max_delta_step, reg_lambda, and min_child_weight prevent Newton-step softmax overflow.
    """
    if task == "binary":
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.08,
            num_leaves=63,
            max_delta_step=2.0,
            reg_lambda=1.0,
            min_child_weight=1e-2,
            subsample_for_bin=200_000,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_delta_step=2.0,
        reg_lambda=10.0,
        min_child_weight=1.0,
        subsample_for_bin=200_000,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )


class FixedWeightEnsemble(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble of Random Forest and LightGBM using fixed scalar weights.

    Combines predicted class probabilities via:
        P_ens(y | x) = w_rf * P_rf(y | x) + w_lgbm * P_lgbm(y | x)
    where w_rf + w_lgbm = 1.
    """

    def __init__(
        self,
        rf=None,
        lgbm=None,
        weights: tuple[float, float] = (0.5, 0.5),
        task: str = "multiclass",
        random_state: int = 42,
        prefitted: bool = False,
    ):
        self.rf = rf
        self.lgbm = lgbm
        self.weights = weights
        self.task = task
        self.random_state = random_state
        self.prefitted = prefitted

    def _normalized_weights(self) -> tuple[float, float]:
        w_rf, w_lgbm = float(self.weights[0]), float(self.weights[1])
        total = w_rf + w_lgbm
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return w_rf / total, w_lgbm / total

    def fit(self, X, y):
        w_rf, w_lgbm = self._normalized_weights()
        X_f32 = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)

        if self.prefitted and self.rf is not None and self.lgbm is not None:
            self.rf_ = self.rf
            self.lgbm_ = self.lgbm
        else:
            rf_base = self.rf if self.rf is not None else default_rf(self.task, self.random_state)
            lgbm_base = self.lgbm if self.lgbm is not None else default_lgbm(self.task, self.random_state)
            self.rf_ = clone(rf_base).fit(X_f32, y)
            self.lgbm_ = clone(lgbm_base).fit(X_f32, y)

        self.classes_ = np.asarray(self.rf_.classes_)
        self.weights_ = (w_rf, w_lgbm)

        # Combine normalized feature importances so eval_template.get_top_features works directly
        rf_imp = _normalize_importances(self.rf_.feature_importances_)
        lgbm_imp = _normalize_importances(self.lgbm_.feature_importances_)
        self.feature_importances_ = w_rf * rf_imp + w_lgbm * lgbm_imp
        return self

    @classmethod
    def from_prefitted(cls, rf_fitted, lgbm_fitted, weights: tuple[float, float] = (0.5, 0.5), task: str = "multiclass"):
        """Constructs a ready-to-predict FixedWeightEnsemble from already-fitted RF and LightGBM models."""
        inst = cls(rf=rf_fitted, lgbm=lgbm_fitted, weights=weights, task=task, prefitted=True)
        w_rf, w_lgbm = inst._normalized_weights()
        inst.rf_ = rf_fitted
        inst.lgbm_ = lgbm_fitted
        inst.classes_ = np.asarray(rf_fitted.classes_)
        inst.weights_ = (w_rf, w_lgbm)
        rf_imp = _normalize_importances(rf_fitted.feature_importances_)
        lgbm_imp = _normalize_importances(lgbm_fitted.feature_importances_)
        inst.feature_importances_ = w_rf * rf_imp + w_lgbm * lgbm_imp
        return inst

    def predict_proba_from_base(self, p_rf: np.ndarray, p_lgbm: np.ndarray) -> np.ndarray:
        """Combines pre-computed probability arrays from RF and LightGBM."""
        w_rf, w_lgbm = self.weights_
        return w_rf * p_rf + w_lgbm * p_lgbm

    def predict_proba(self, X) -> np.ndarray:
        X_f32 = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        p_rf = self.rf_.predict_proba(X_f32)
        p_lgbm = self.lgbm_.predict_proba(X_f32)
        return self.predict_proba_from_base(p_rf, p_lgbm)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def build_meta_features(p_rf: np.ndarray, p_lgbm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Constructs stacking meta-features from RF and LightGBM probability distributions.

    Includes:
    - Raw class probabilities from RF and LightGBM in [0, 1] (mixture-of-experts)
    - Scaled log-probabilities log(clip(P, eps, 1)) / log(1/eps) in [-1, 0] (product-of-experts)
    - Per-class absolute probability difference |P_rf - P_lgbm| (disagreement signal)
    """
    p_rf_c = np.clip(p_rf, eps, 1.0 - eps)
    p_lgbm_c = np.clip(p_lgbm, eps, 1.0 - eps)
    scale = float(np.log(1.0 / eps))
    log_rf = np.log(p_rf_c) / scale
    log_lgbm = np.log(p_lgbm_c) / scale
    diff = np.abs(p_rf - p_lgbm)
    return np.hstack([p_rf, p_lgbm, log_rf, log_lgbm, diff]).astype(np.float32)


def _subsample_for_meta(
    meta_X: np.ndarray,
    y: np.ndarray,
    max_per_class: int = 8000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retains 100% of minority-class rows and caps majority classes at max_per_class with exact importance weights."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    idx_list = []
    weights_list = []

    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        if cnt > max_per_class:
            chosen = rng.choice(c_idx, size=max_per_class, replace=False)
            w = float(cnt) / float(max_per_class)
        else:
            chosen = c_idx
            w = 1.0
        idx_list.append(chosen)
        weights_list.append(np.full(len(chosen), w, dtype=np.float32))

    all_idx = np.concatenate(idx_list)
    all_w = np.concatenate(weights_list)
    order = np.argsort(all_idx)
    return meta_X[all_idx[order]], y[all_idx[order]], all_w[order]


def _fit_class_multipliers(
    probs: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    max_samples: int = 60000,
    random_state: int = 42,
) -> np.ndarray:
    """Coordinate-ascent calibration of per-class probability multipliers to maximize Macro-F1."""
    k = len(classes)
    mult = np.ones(k, dtype=np.float32)
    if len(y_true) == 0 or k < 2:
        return mult

    # Use a stratified subsample if validation set is huge so coordinate search takes <0.5s
    if len(y_true) > max_samples:
        probs_sub, y_sub, _ = _subsample_for_meta(probs, y_true, max_per_class=max_samples // k, random_state=random_state)
    else:
        probs_sub, y_sub = probs, y_true

    class_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([class_to_int[val] for val in y_sub], dtype=np.int32)
    labels_int = np.arange(k, dtype=np.int32)

    best_f1 = float(f1_score(y_int, np.argmax(probs_sub * mult, axis=1), labels=labels_int, average="macro", zero_division=0))
    candidates = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0, 4.5]

    for _ in range(2):
        for c_i in range(k):
            best_m = mult[c_i]
            for cand in candidates:
                mult[c_i] = cand
                preds = np.argmax(probs_sub * mult, axis=1)
                score = float(f1_score(y_int, preds, labels=labels_int, average="macro", zero_division=0))
                if score > best_f1 + 1e-5:
                    best_f1 = score
                    best_m = cand
            mult[c_i] = best_m
    return mult


class MetaLearnerEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble combining Random Forest and LightGBM via a trained Meta-Learner.

    Unlike fixed scalar weights, the meta-learner learns class-specific weights over both
    raw probabilities and log-probabilities from RF and LightGBM, plus per-class decision
    multipliers calibrated on out-of-sample predictions to maximize Macro-F1.
    """

    def __init__(
        self,
        rf=None,
        lgbm=None,
        meta_learner=None,
        task: str = "multiclass",
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.rf = rf
        self.lgbm = lgbm
        self.meta_learner = meta_learner
        self.task = task
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _make_meta_learner(self):
        if self.meta_learner is not None:
            return clone(self.meta_learner)
        return LogisticRegression(
            C=20.0,
            max_iter=500,
            random_state=self.random_state,
            solver="lbfgs",
        )

    def _fit_meta_on_oof(self, p_rf: np.ndarray, p_lgbm: np.ndarray, y_arr: np.ndarray):
        """Fits the stacking LogisticRegression and per-class Macro-F1 multipliers on out-of-sample predictions."""
        meta_X = build_meta_features(p_rf, p_lgbm)
        meta_X_sub, y_sub, sample_w = _subsample_for_meta(
            meta_X, y_arr, max_per_class=8000, random_state=self.random_state
        )
        est = self._make_meta_learner()
        try:
            est.fit(meta_X_sub, y_sub, sample_weight=sample_w)
        except TypeError:
            est.fit(meta_X_sub, y_sub)
        self.meta_learner_ = est

        # Blend stacking model probabilities with base log-geometric/arithmetic mean and calibrate class multipliers
        raw_meta_probs = self.meta_learner_.predict_proba(meta_X_sub)
        p_rf_sub, _, _ = _subsample_for_meta(p_rf, y_arr, max_per_class=8000, random_state=self.random_state)
        p_lgbm_sub, _, _ = _subsample_for_meta(p_lgbm, y_arr, max_per_class=8000, random_state=self.random_state)
        blended_sub = 0.5 * raw_meta_probs + 0.25 * p_rf_sub + 0.25 * p_lgbm_sub
        self.class_multipliers_ = _fit_class_multipliers(
            blended_sub, y_sub, self.classes_, max_samples=40000, random_state=self.random_state
        )

    def _compute_feature_importances(self):
        """Derives ensemble feature importances weighted by the meta-learner's reliance on RF vs. LightGBM."""
        rf_imp = _normalize_importances(self.rf_.feature_importances_)
        lgbm_imp = _normalize_importances(self.lgbm_.feature_importances_)
        contrib = self.get_model_contributions()
        w_rf = contrib["rf_share"]
        w_lgbm = contrib["lgbm_share"]
        self.feature_importances_ = w_rf * rf_imp + w_lgbm * lgbm_imp

    def get_model_contributions(self) -> dict[str, float]:
        """Quantifies the relative weight assigned to RF vs. LightGBM by the fitted meta-learner."""
        est = self.meta_learner_.steps[-1][1] if hasattr(self.meta_learner_, "steps") else self.meta_learner_
        if not hasattr(est, "coef_"):
            return {"rf_share": 0.5, "lgbm_share": 0.5}

        coef = np.abs(est.coef_)
        k = len(self.classes_)
        if coef.shape[1] >= 4 * k:
            rf_mag = float(np.sum(coef[:, :k]) + np.sum(coef[:, 2 * k : 3 * k]))
            lgbm_mag = float(np.sum(coef[:, k : 2 * k]) + np.sum(coef[:, 3 * k : 4 * k]))
            total = rf_mag + lgbm_mag
            if total > 0:
                return {
                    "rf_share": round(rf_mag / total, 4),
                    "lgbm_share": round(lgbm_mag / total, 4),
                }
        return {"rf_share": 0.5, "lgbm_share": 0.5}

    def fit(self, X, y):
        """Fits base models and trains the meta-learner using out-of-fold probability predictions."""
        X_arr = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        X_np = X_arr.to_numpy(dtype=np.float32) if isinstance(X_arr, pd.DataFrame) else np.asarray(X_arr, dtype=np.float32)
        y_arr = np.asarray(y)

        rf_base = self.rf if self.rf is not None else default_rf(self.task, self.random_state)
        lgbm_base = self.lgbm if self.lgbm is not None else default_lgbm(self.task, self.random_state)

        classes = np.unique(y_arr)
        self.classes_ = classes
        n_classes = len(classes)

        _, counts = np.unique(y_arr, return_counts=True)
        n_splits = min(self.cv_folds, int(counts.min()))

        oof_rf = np.zeros((len(y_arr), n_classes), dtype=np.float32)
        oof_lgbm = np.zeros((len(y_arr), n_classes), dtype=np.float32)

        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            for tr_idx, va_idx in skf.split(X_np, y_arr):
                if len(np.unique(y_arr[tr_idx])) < n_classes:
                    continue
                rf_fold = clone(rf_base).fit(X_np[tr_idx], y_arr[tr_idx])
                lgbm_fold = clone(lgbm_base).fit(X_np[tr_idx], y_arr[tr_idx])
                oof_rf[va_idx] = rf_fold.predict_proba(X_np[va_idx])
                oof_lgbm[va_idx] = lgbm_fold.predict_proba(X_np[va_idx])

        self.rf_ = clone(rf_base).fit(X_arr, y_arr)
        self.lgbm_ = clone(lgbm_base).fit(X_arr, y_arr)

        missing = (oof_rf.sum(axis=1) == 0)
        if missing.any():
            sub_X = X_arr.iloc[np.where(missing)[0]] if isinstance(X_arr, pd.DataFrame) else X_arr[missing]
            oof_rf[missing] = self.rf_.predict_proba(sub_X)
            oof_lgbm[missing] = self.lgbm_.predict_proba(sub_X)

        self._fit_meta_on_oof(oof_rf, oof_lgbm, y_arr)
        self._compute_feature_importances()
        return self

    @classmethod
    def from_prefitted(
        cls,
        rf_fitted,
        lgbm_fitted,
        X_val,
        y_val,
        task: str = "multiclass",
        meta_learner=None,
        random_state: int = 42,
        p_rf_val: np.ndarray | None = None,
        p_lgbm_val: np.ndarray | None = None,
    ):
        """Constructs and fits a MetaLearnerEnsemble using pre-fitted base models and held-out validation data."""
        inst = cls(rf=rf_fitted, lgbm=lgbm_fitted, meta_learner=meta_learner, task=task, random_state=random_state)
        inst.rf_ = rf_fitted
        inst.lgbm_ = lgbm_fitted
        inst.classes_ = np.asarray(rf_fitted.classes_)

        if p_rf_val is None or p_lgbm_val is None:
            X_f32 = X_val.astype(np.float32) if hasattr(X_val, "astype") else np.asarray(X_val, dtype=np.float32)
            p_rf_val = rf_fitted.predict_proba(X_f32)
            p_lgbm_val = lgbm_fitted.predict_proba(X_f32)

        inst._fit_meta_on_oof(p_rf_val, p_lgbm_val, np.asarray(y_val))
        inst._compute_feature_importances()
        return inst

    def predict_proba_from_base(self, p_rf: np.ndarray, p_lgbm: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probabilities from pre-computed base model probability arrays."""
        meta_X = build_meta_features(p_rf, p_lgbm)
        raw_meta = self.meta_learner_.predict_proba(meta_X)
        blended = 0.5 * raw_meta + 0.25 * p_rf + 0.25 * p_lgbm
        scaled = blended * self.class_multipliers_
        row_sums = np.sum(scaled, axis=1, keepdims=True)
        return np.where(row_sums > 0, scaled / row_sums, blended)

    def predict_proba(self, X) -> np.ndarray:
        X_f32 = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        p_rf = self.rf_.predict_proba(X_f32)
        p_lgbm = self.lgbm_.predict_proba(X_f32)
        return self.predict_proba_from_base(p_rf, p_lgbm)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def sweep_fixed_weights(
    p_rf: np.ndarray,
    p_lgbm: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    benign_name: str,
    steps: int = 21,
) -> dict:
    """Evaluates a grid of fixed weights w_rf in [0, 1] (with w_lgbm = 1 - w_rf) and finds the optimal weight."""
    grid = []
    best_entry = None
    classes_arr = np.asarray(classes)
    class_to_idx = {c: i for i, c in enumerate(classes_arr)}
    y_int = np.array([class_to_idx[v] for v in y_true], dtype=np.int32)
    benign_idx = class_to_idx.get(benign_name, -1)
    is_benign = (y_int == benign_idx)
    labels_int = np.arange(len(classes_arr), dtype=np.int32)

    for w_rf in np.linspace(0.0, 1.0, steps):
        w_rf = round(float(w_rf), 4)
        w_lgbm = round(1.0 - w_rf, 4)
        p_comb = w_rf * p_rf + w_lgbm * p_lgbm
        preds_int = np.argmax(p_comb, axis=1)
        f1 = float(f1_score(y_int, preds_int, labels=labels_int, average="macro", zero_division=0))
        fpr = float((preds_int[is_benign] != benign_idx).mean()) if is_benign.any() else 0.0
        entry = {
            "w_rf": w_rf,
            "w_lgbm": w_lgbm,
            "macro_f1": round(f1, 5),
            "benign_fpr": round(fpr, 5),
        }
        grid.append(entry)
        if best_entry is None or f1 > best_entry["macro_f1"] or (
            abs(f1 - best_entry["macro_f1"]) < 1e-6 and fpr < best_entry["benign_fpr"]
        ):
            best_entry = entry

    return {
        "best_weights": (best_entry["w_rf"], best_entry["w_lgbm"]),
        "best_macro_f1": best_entry["macro_f1"],
        "best_benign_fpr": best_entry["benign_fpr"],
        "grid": grid,
    }

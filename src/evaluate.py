"""
Cross-validated evaluation of the full pipeline.

The whole pipeline (Stage 1, autoencoder, Stage 2) is refitted inside every
fold, and every metric is computed on households the fold's model never saw.
Folds are stratified on the binary outcome so each fold keeps its share of
non-enrolled households.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config as C
from src.models import (LATENT_COLUMNS, TARGETS, fit_pipeline, make_stage2, predict,
                        stage1_probabilities, stage2_frame)


def _binary(y):
    return (np.asarray(y) >= C.ENROLLED_THRESHOLD).astype(int)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    yb, pb = _binary(y_true), _binary(y_pred)
    both = len(np.unique(yb)) == 2
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "auc": float(roc_auc_score(yb, y_pred)) if both else None,
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(yb, pb)) if both else None,
        "not_enrolled_found_at_0.5": int(((pb == 0) & (yb == 0)).sum()),
        "not_enrolled_total": int((yb == 0).sum()),
        "mean_predicted": float(y_pred.mean()),
        "mean_actual": float(y_true.mean()),
    }


def cross_validate(df: pd.DataFrame, preset: str) -> dict:
    df = df.reset_index(drop=True)
    y = df[C.TARGET_COLUMN].to_numpy(float)
    yb = _binary(y)
    # Every fold needs at least one non-enrolled household (urban has very few).
    n_splits = max(2, min(C.CV_FOLDS, int(np.bincount(yb, minlength=2).min())))
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=C.CV_REPEATS, random_state=C.SEED)

    variants = ["full_model", "without_perceptions", "xgboost_only", "random_forest_only",
                "linear_only", "mean_baseline"]
    oof = {r: {v: np.full(len(df), np.nan) for v in variants} for r in range(C.CV_REPEATS)}
    stage1_true = {t: [] for t in TARGETS}
    stage1_pred = {t: [] for t in TARGETS}
    stage1_score = {t: [] for t in TARGETS}
    stage1_majority = {t: [] for t in TARGETS}

    for fold, (tr, te) in enumerate(splitter.split(df, yb)):
        repeat = fold // n_splits
        train, test = df.iloc[tr], df.iloc[te]
        bundle, s2_train = fit_pipeline(train, preset, seed=C.SEED + fold, return_stage2_inputs=True)
        out = predict(bundle, test)
        oof[repeat]["full_model"][te] = out["model_prob"].to_numpy()

        # Stage 1 on held-out households
        probs = stage1_probabilities(bundle["stage1"], test[C.FEATURE_COLUMNS].to_numpy(np.float32))
        for t in TARGETS:
            truth = test[t].to_numpy(int)
            stage1_true[t].append(truth)
            stage1_pred[t].append(probs[t].argmax(axis=1))
            stage1_score[t].append(out[t].to_numpy())
            majority = np.bincount(train[t].to_numpy(int), minlength=C.N_LIKERT_CLASSES).argmax()
            stage1_majority[t].append(np.full(len(te), majority))

        # Comparison models on identical folds and identical Stage 2 inputs
        y_train = train[C.TARGET_COLUMN].to_numpy(float)
        s2_test = stage2_frame(test, out[LATENT_COLUMNS].to_numpy())
        for name, est in make_stage2(C.SEED + fold).estimators:
            m = clone(est).fit(s2_train, y_train)
            oof[repeat][f"{name}_only"][te] = np.clip(m.predict(s2_test), 0, 1)
        factual = C.STAGE2_FACTUAL_COLUMNS
        m = make_stage2(C.SEED + fold).fit(train[factual], y_train)
        oof[repeat]["without_perceptions"][te] = np.clip(m.predict(test[factual]), 0, 1)
        oof[repeat]["mean_baseline"][te] = y_train.mean()

    def _average(y_true, preds_by_repeat):
        runs = [regression_metrics(y_true, p) for p in preds_by_repeat]
        return {k: (None if runs[0][k] is None else float(np.mean([m[k] for m in runs])))
                for k in runs[0]}

    stage2 = {v: _average(y, [oof[r][v] for r in range(C.CV_REPEATS)]) for v in variants}
    # Robustness: score only households whose school answers were observed,
    # not copied from a neighbour (see README, Data).
    observed = df["school_info_imputed"].to_numpy() == 0
    if len(np.unique(yb[observed])) == 2:
        stage2["full_model_observed_answers_only"] = _average(
            y[observed], [oof[r]["full_model"][observed] for r in range(C.CV_REPEATS)])

    stage1 = {}
    for t in TARGETS:
        truth, pred = np.concatenate(stage1_true[t]), np.concatenate(stage1_pred[t])
        stage1[t] = {
            "survey_item": C.STAGE1_TARGETS[t],
            "accuracy": float(accuracy_score(truth, pred)),
            "majority_class_accuracy": float(accuracy_score(truth, np.concatenate(stage1_majority[t]))),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro")),
            "spearman_expected_score": float(spearmanr(truth, np.concatenate(stage1_score[t])).correlation),
            "answer_distribution": np.bincount(truth, minlength=C.N_LIKERT_CLASSES).tolist(),
        }

    return {
        "preset": preset,
        "households": len(df),
        "not_enrolled_households": int((yb == 0).sum()),
        "folds": n_splits,
        "repeats": C.CV_REPEATS,
        "stage1": stage1,
        "stage2": stage2,
    }

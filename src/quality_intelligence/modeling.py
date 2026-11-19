"""Training, calibration and operational policy evaluation."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from quality_intelligence.data import FEATURES


def metrics(y, p, threshold=0.5) -> dict:
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "inspection_burden": float(pred.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def policies(y, p, fn_cost=15, fp_cost=1) -> dict:
    rows = [(float(t), metrics(y, p, t)) for t in np.linspace(0, 1, 201)]
    high = [r for r in rows if r[1]["recall"] >= 0.9]
    low = [r for r in rows if r[1]["false_positive_rate"] <= 0.02]
    return {
        "high_recall": min(high, key=lambda r: r[1]["inspection_burden"])[0],
        "low_false_reject": max(low, key=lambda r: r[1]["recall"])[0],
        "balanced_cost": min(
            rows,
            key=lambda r: (
                fn_cost * r[1]["confusion_matrix"][1][0] + fp_cost * r[1]["confusion_matrix"][0][1]
            ),
        )[0],
    }


def train(train: pd.DataFrame, tune: pd.DataFrame, seed=42) -> tuple[dict, list]:
    if train.defect.nunique() != 2 or tune.defect.nunique() != 2:
        raise ValueError("Training and tuning require both classes")
    factories = {
        "logistic": lambda size: LogisticRegression(C=size, class_weight="balanced", max_iter=1000),
        "random_forest": lambda size: RandomForestClassifier(
            n_estimators=120,
            max_depth=int(size),
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=2,
            random_state=seed,
        ),
        "xgboost": lambda size: XGBClassifier(
            n_estimators=160,
            max_depth=int(size),
            learning_rate=0.06,
            n_jobs=2,
            random_state=seed,
            eval_metric="logloss",
        ),
        "lightgbm": lambda size: LGBMClassifier(
            n_estimators=160,
            num_leaves=int(size),
            learning_rate=0.05,
            verbosity=-1,
            n_jobs=2,
            random_state=seed,
        ),
    }
    grids = {
        "logistic": [0.1, 1.0],
        "random_forest": [6, 12],
        "xgboost": [2, 4],
        "lightgbm": [7, 15],
    }
    models, trials = {}, []
    for name, factory in factories.items():
        best = -1
        for size in grids[name]:
            model = make_pipeline(SimpleImputer(), StandardScaler(), factory(size))
            model.fit(train[FEATURES], train.defect)
            score = average_precision_score(tune.defect, model.predict_proba(tune[FEATURES])[:, 1])
            trials.append({"model": name, "parameter": size, "validation_pr_auc": score})
            if score > best:
                models[name], best = model, score
    return models, trials


def calibrate(model, calibration, policy) -> tuple[dict, list]:
    if calibration.defect.nunique() < 2 or policy.defect.nunique() < 2:
        raise ValueError("Calibration and policy sets require both classes")
    pc = model.predict_proba(calibration[FEATURES])[:, 1]
    pp = model.predict_proba(policy[FEATURES])[:, 1]
    sigmoid = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), calibration.defect)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(pc, calibration.defect)
    options = {"uncalibrated": None, "sigmoid": sigmoid, "isotonic": isotonic}
    scores = [
        {"method": name, "policy_brier": brier_score_loss(policy.defect, transform(obj, pp))}
        for name, obj in options.items()
    ]
    selected = min(scores, key=lambda r: r["policy_brier"])["method"]
    return {"model": model, "calibrator": options[selected], "method": selected}, scores


def transform(calibrator, p):
    if calibrator is None:
        return p
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(p)
    return calibrator.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def predict(bundle, frame):
    return transform(bundle["calibrator"], bundle["model"].predict_proba(frame[FEATURES])[:, 1])


def timed_prediction(bundle, frame):
    start = perf_counter()
    p = predict(bundle, frame)
    return p, (perf_counter() - start) * 1000 / len(frame)


def anomaly(train, seed=42):
    model = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        IsolationForest(n_estimators=150, contamination=0.06, random_state=seed, n_jobs=2),
    )
    return model.fit(train[FEATURES])

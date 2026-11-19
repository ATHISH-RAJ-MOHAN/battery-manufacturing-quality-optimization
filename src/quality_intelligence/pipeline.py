"""Single reproducible orchestration entry point."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quality_intelligence.analysis import explain, investigate, optimize
from quality_intelligence.data import FEATURES, generate, split, validate
from quality_intelligence.modeling import (
    anomaly,
    calibrate,
    metrics,
    policies,
    predict,
    timed_prediction,
    train,
    transform,
)

LOG = logging.getLogger(__name__)


def run(
    output: Path, n=24000, seed=42, rate=0.06, batch_size=200, fn_cost=15, fp_cost=1, quick=False
) -> dict:
    output = Path(output)
    for directory in ["data", "models", "reports"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    frame, intercept = generate(n, seed, rate, batch_size)
    quality = validate(frame)
    path = output / "data/modules.parquet"
    frame.to_parquet(path, index=False)
    fit, tune, calibration, policy, test = split(frame)
    LOG.info("Training on %s rows; held-out test %s", len(fit), len(test))
    models, trials = train(fit, tune, seed)
    # Select family on tuning data, before touching test results.
    family = max(trials, key=lambda x: x["validation_pr_auc"])["model"]
    bundles, rows, calibration_rows = {}, [], []
    predictions = test[["module_id", "defect", "batch", "equipment"]].copy()
    for name, model in models.items():
        bundle, scores = calibrate(model, calibration, policy)
        bundle["policies"] = policies(policy.defect, predict(bundle, policy), fn_cost, fp_cost)
        bundles[name] = bundle
        probability, latency = timed_prediction(bundle, test)
        predictions[name] = probability
        row = {
            "model": name,
            **metrics(test.defect, probability, bundle["policies"]["balanced_cost"]),
            "latency_ms_per_row": latency,
            "calibration": bundle["method"],
        }
        rows.append(row)
        raw = model.predict_proba(test[FEATURES])[:, 1]
        for score in scores:
            calibration_rows.append(
                {
                    "model": name,
                    **score,
                    "test_brier": metrics(
                        test.defect,
                        transform(
                            None
                            if score["method"] == "uncalibrated"
                            # Re-fit each comparison only on calibration partition.
                            else _calibrator(score["method"], model, calibration),
                            raw,
                        ),
                    )["brier"],
                }
            )
    chosen = bundles[family]
    detector = anomaly(fit, seed)
    anomaly_score = -detector.decision_function(test[FEATURES])
    alert = detector.predict(test[FEATURES]) == -1
    predictions["anomaly_score"] = anomaly_score
    predictions["anomaly_alert"] = alert
    anomaly_metrics = metrics(test.defect, alert.astype(float))
    sample = test.sample(min(160, len(test)), random_state=seed)
    shap_values = explain(chosen["model"], sample)
    importance = dict(zip(FEATURES, np.abs(shap_values).mean(axis=0).tolist()))
    ranked = test.assign(risk=predictions[family].to_numpy()).groupby("batch").risk.mean()
    batch_values = explain(chosen["model"], test[test.batch == ranked.idxmax()])
    batch_importance = dict(
        sorted(
            zip(FEATURES, np.abs(batch_values).mean(axis=0).tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    batches, report = investigate(
        test, predictions[family].to_numpy(), importance, batch_importance
    )
    optimization = optimize(
        chosen,
        test[FEATURES].fillna(fit[FEATURES].median()).sample(min(40, len(test)), random_state=seed),
        intercept,
        budget=25 if quick else 120,
    )
    optimization.to_csv(output / "reports/optimization.csv", index=False)
    batches.to_csv(output / "reports/batches.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "reports/tuning.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "reports/calibration.csv", index=False)
    predictions.to_parquet(output / "reports/predictions.parquet", index=False)
    sample.to_parquet(output / "reports/explanation_sample.parquet", index=False)
    np.save(output / "reports/shap.npy", shap_values)
    pd.Series(importance, name="mean_absolute_shap").sort_values().to_csv(
        output / "reports/importance.csv"
    )
    (output / "reports/investigation.md").write_text(report, encoding="utf-8")
    joblib.dump(
        {
            "selected": family,
            "bundle": chosen,
            "anomaly": detector,
            "defaults": fit[FEATURES].median().to_dict(),
        },
        output / "models/system.joblib",
    )
    result = {
        "seed": seed,
        "configuration": {
            "records": n,
            "defect_rate": rate,
            "batch_size": batch_size,
            "false_negative_cost": fn_cost,
            "false_positive_cost": fp_cost,
        },
        "data_quality": quality,
        "selected_model": family,
        "models": rows,
        "anomaly": anomaly_metrics,
        "policies": chosen["policies"],
        "policy_test_metrics": {
            k: metrics(test.defect, predictions[family], t) for k, t in chosen["policies"].items()
        },
        "split_rows": [len(s) for s in [fit, tune, calibration, policy, test]],
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "shap", "xgboost"]},
    }
    (output / "reports/results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows).drop(columns="confusion_matrix").to_string(index=False)
    summary = optimization.groupby("method").oracle_probability.agg(["mean", "std", "count"])
    summary["approx_95ci_halfwidth"] = 4.303 * summary["std"] / np.sqrt(summary["count"])
    (output / "reports/evaluation.md").write_text(
        f"# Verified synthetic experiment\n\nSeed {seed}; {n} rows; selected {family} on tuning PR-AUC.\n\n"
        f"```\n{table}\n```\n\nAnomaly metrics:\n```\n{anomaly_metrics}\n```\n\n"
        f"Independent simulator probability across three search seeds:\n```\n{summary.to_string()}\n```\n\n"
        "Intervals describe search variability for fixed contexts and simulator, not factory uncertainty. "
        "Constrained search includes random initialization cost in addition to reported solver time. "
        "Oracle evaluates expected probabilities without sampled-label noise; it is separate from "
        "the surrogate but shares the training simulator assumptions. No physical quality improvement is established.\n",
        encoding="utf-8",
    )
    LOG.info("Finished; results at %s", output / "reports/results.json")
    return result


def _calibrator(method, model, data):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    p = model.predict_proba(data[FEATURES])[:, 1]
    if method == "sigmoid":
        return LogisticRegression(C=1e6).fit(p.reshape(-1, 1), data.defect)
    return IsotonicRegression(out_of_bounds="clip").fit(p, data.defect)

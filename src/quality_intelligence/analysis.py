"""Investigation hypotheses and independently scored optimization."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quality_intelligence.data import BOUNDS, CONTROLS, FEATURES, oracle
from quality_intelligence.modeling import predict


def explain(model, frame):
    import shap

    transformed = model[:-1].transform(frame[FEATURES])
    estimator = model[-1]
    if hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, transformed[:100])
    else:
        explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != (len(frame), len(FEATURES)) or not np.isfinite(values).all():
        raise ValueError("Invalid SHAP explanation")
    return values


def investigate(
    frame, probabilities, importance: dict, batch_contributors=None
) -> tuple[pd.DataFrame, str]:
    data = frame.assign(predicted_risk=probabilities)
    summary = data.groupby("batch").agg(
        modules=("defect", "size"),
        observed_rate=("defect", "mean"),
        predicted_risk=("predicted_risk", "mean"),
    )
    summary = summary.sort_values("predicted_risk", ascending=False)
    batch = summary.index[0]
    target = data[data.batch == batch]
    reference = data[data.batch != batch]
    deviations = (
        (
            (target[FEATURES].mean() - reference[FEATURES].mean())
            / reference[FEATURES].std().replace(0, np.nan)
        )
        .abs()
        .sort_values(ascending=False)
    )
    equipment = target.groupby("equipment").defect.agg(["mean", "size"]).to_string()
    report = (
        f"# Batch investigation: {batch}\n\nSynthetic observations only. "
        "Associations are hypotheses, not confirmed causes.\n\n"
        f"Observed defect rate: {target.defect.mean():.3%}; n={len(target)}. "
        f"Reference rate: {reference.defect.mean():.3%}. Small batch rates are noisy.\n\n"
        f"Largest absolute standardized mean deviations:\n\n{deviations.head().to_string()}\n\n"
        f"Equipment observations (rate, count):\n\n```\n{equipment}\n```\n\n"
        f"Global model SHAP contributors: {importance}. These explain raw model scores, "
        f"not calibrated probabilities. Batch-specific mean absolute SHAP: {batch_contributors}. "
        "Inspect sensors, recipe logs and maintenance history; "
        "confirm hypotheses with controlled engineering studies before any intervention.\n"
    )
    return summary.reset_index(), report


def feasible(x) -> bool:
    return (
        all(a <= v <= b for v, (a, b) in zip(x, BOUNDS, strict=True)) and 3.5 <= x[0] / x[1] <= 6.5
    )


def optimize(bundle, reference, intercept, seeds=(11, 22, 33), budget=120) -> pd.DataFrame:
    """Optimize common controls on context rows; oracle is evaluation-only."""
    if reference.empty or budget < 5:
        raise ValueError("Optimization requires context rows and budget >=5")
    results = []
    baseline = np.array([56.0, 9.0, 5.7, 33.0])

    def apply(x):
        candidate = reference.copy()
        candidate[CONTROLS] = np.tile(x, (len(candidate), 1))
        return candidate

    def objective(x):
        return float(predict(bundle, apply(x)).mean())

    for seed in seeds:
        rng = np.random.default_rng(seed)
        start = perf_counter()
        candidates = rng.uniform(np.array(BOUNDS)[:, 0], np.array(BOUNDS)[:, 1], (budget, 4))
        candidates = [x for x in candidates if feasible(x)]
        random_best = min(candidates, key=objective)
        random_time = perf_counter() - start
        start = perf_counter()
        solution = minimize(
            objective,
            random_best,
            method="COBYLA",
            bounds=BOUNDS,
            constraints=[
                {"type": "ineq", "fun": lambda x: x[0] - 3.5 * x[1]},
                {"type": "ineq", "fun": lambda x: 6.5 * x[1] - x[0]},
            ],
            options={"maxiter": budget, "rhobeg": 1.0},
        )
        chosen = solution.x if feasible(solution.x) else random_best
        constrained_time = perf_counter() - start
        for method, x, elapsed in [
            ("baseline", baseline, 0),
            ("random", random_best, random_time),
            ("constrained", chosen, constrained_time),
        ]:
            candidate = apply(x)
            results.append(
                {
                    "seed": seed,
                    "method": method,
                    **dict(zip(CONTROLS, map(float, x))),
                    "predicted_risk": objective(x),
                    "oracle_probability": float(oracle(candidate, intercept).mean()),
                    "constraint_violations": int(not feasible(x)),
                    "seconds": elapsed,
                    "solver_success": bool(solution.success) if method == "constrained" else True,
                    "solver_evaluations": int(solution.nfev)
                    if method == "constrained"
                    else len(candidates)
                    if method == "random"
                    else 0,
                }
            )
    return pd.DataFrame(results)

"""Synthetic assumptions are illustrative, never physical battery specifications."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

CENTERS = {
    "voltage_variation": 8.0,
    "internal_resistance": 18.0,
    "welding_resistance": 1.2,
    "current": 100.0,
    "pretest_voltage": 48.0,
    "process_temperature": 25.0,
    "interface_temperature": 30.0,
    "temperature_variation": 2.0,
    "welding_energy": 50.0,
    "welding_duration": 10.0,
    "assembly_pressure": 5.0,
    "alignment_deviation": 0.0,
    "cycle_time": 60.0,
}
SCALES = dict(zip(CENTERS, [2, 2, 0.2, 5, 0.3, 3, 3, 0.5, 7, 1.2, 0.7, 0.2, 5]))
FEATURES = list(CENTERS)
CONTROLS = ["welding_energy", "welding_duration", "assembly_pressure", "interface_temperature"]
BOUNDS = [(35.0, 65.0), (7.0, 13.0), (3.5, 6.5), (24.0, 36.0)]


def severity(frame: pd.DataFrame) -> np.ndarray:
    """Structural simulator: nonlinear deviations and coupled weld power."""
    z = (frame[FEATURES] - pd.Series(CENTERS)) / pd.Series(SCALES)
    return (
        0.7 * z.welding_energy**2
        + 0.5 * z.assembly_pressure**2
        + 0.35 * z.interface_temperature**2
        + 0.7 * z.alignment_deviation.abs()
        + 0.5 * z.voltage_variation
        + 0.4 * z.internal_resistance
        + 0.6 * ((frame.welding_energy / frame.welding_duration - 5) / 1.1) ** 2
        + 0.3 * z.temperature_variation
    ).to_numpy()


def oracle(frame: pd.DataFrame, intercept: float) -> np.ndarray:
    """Expected defect probability; never a model input or training target."""
    return expit(intercept + severity(frame))


def generate(
    n: int = 24000, seed: int = 42, rate: float = 0.06, batch_size: int = 200
) -> tuple[pd.DataFrame, float]:
    if n < 100 or not 0 < rate < 0.5 or batch_size < 10:
        raise ValueError("Need >=100 rows, batch size >=10 and 0<rate<0.5")
    rng = np.random.default_rng(seed)
    batch = np.arange(n) // batch_size
    equipment = rng.integers(0, 8, n)
    batch_effect = rng.normal(0, 0.25, batch.max() + 1)[batch]
    common = rng.normal(0, 0.4, n) + batch_effect + (equipment - 3.5) * 0.07
    frame = pd.DataFrame(
        {k: v + SCALES[k] * (rng.normal(0, 0.8, n) + common) for k, v in CENTERS.items()}
    )
    frame["welding_duration"] += (frame.welding_energy - 50) * 0.04
    intercept = brentq(lambda b: oracle(frame, b).mean() - rate, -40, 5)
    probability = oracle(frame, intercept)
    frame["module_id"] = [f"M{i:07d}" for i in range(n)]
    frame["batch"] = batch
    frame["equipment"] = [f"EQ-{e}" for e in equipment]
    frame["station"] = equipment % 4
    frame["shift"] = (np.arange(n) // 480) % 3
    frame["timestamp"] = pd.date_range("2025-01-01", periods=n, freq="min")
    frame["defect"] = rng.binomial(1, probability)
    frame["simulated_probability"] = probability
    z = (frame[CONTROLS] - pd.Series(CENTERS)[CONTROLS]).abs()
    frame["defect_category"] = np.where(frame.defect, z.idxmax(axis=1), "pass")
    # Sensor dropout affects recorded features, after latent quality generation.
    for col in FEATURES:
        frame.loc[rng.random(n) < 0.003, col] = np.nan
    return frame, intercept


def validate(frame: pd.DataFrame) -> dict:
    required = FEATURES + ["module_id", "batch", "timestamp", "defect", "equipment"]
    if frame.empty or not set(required).issubset(frame):
        raise ValueError("Empty dataset or missing required columns")
    if frame.module_id.duplicated().any() or frame.duplicated().any():
        raise ValueError("Duplicate module records")
    if not frame.defect.isin([0, 1]).all():
        raise ValueError("Defect must be binary")
    values = frame[FEATURES]
    if any(not pd.api.types.is_numeric_dtype(values[c]) for c in FEATURES):
        raise ValueError("Measurements must be numeric")
    if np.isinf(values.to_numpy()).any() or values.isna().mean().max() > 0.2:
        raise ValueError("Invalid sensor values or excessive missingness")
    return {
        "rows": len(frame),
        "missing": values.isna().sum().to_dict(),
        "defect_rate": float(frame.defect.mean()),
        "duplicates": 0,
        "outliers_over_3_nominal_scales": (
            (values - pd.Series(CENTERS)).abs() > 3 * pd.Series(SCALES)
        )
        .sum()
        .to_dict(),
    }


def split(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Five chronological, batch-disjoint sets: fit/tune/calibrate/policy/test."""
    batches = np.sort(frame.batch.unique())
    if len(batches) < 10:
        raise ValueError("At least ten production batches required")
    cuts = [
        0,
        int(0.5 * len(batches)),
        int(0.65 * len(batches)),
        int(0.75 * len(batches)),
        int(0.85 * len(batches)),
        len(batches),
    ]
    return [frame[frame.batch.isin(batches[a:b])].copy() for a, b in itertools.pairwise(cuts)]

"""Compare seeded data, probabilities and scientific metrics, excluding runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("first", type=Path)
parser.add_argument("second", type=Path)
args = parser.parse_args()
for relative in ["data/modules.parquet", "reports/predictions.parquet"]:
    pd.testing.assert_frame_equal(
        pd.read_parquet(args.first / relative),
        pd.read_parquet(args.second / relative),
        check_exact=relative.startswith("data/"),
        rtol=1e-12,
        atol=1e-12,
    )
a = json.loads((args.first / "reports/results.json").read_text())
b = json.loads((args.second / "reports/results.json").read_text())
for key in ["dataset_sha256", "selected_model", "policies", "policy_test_metrics"]:
    assert a[key] == b[key], key
for left, right in zip(a["models"], b["models"], strict=True):
    for key, value in left.items():
        if key == "latency_ms_per_row":
            continue
        if isinstance(value, float):
            np.testing.assert_allclose(value, right[key], rtol=1e-12, atol=1e-12)
        else:
            assert value == right[key], key
print(
    "MATCH: exact dataset, selection and policies; predictions and metrics within 1e-12 (excluding latency)."
)

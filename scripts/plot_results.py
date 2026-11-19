"""Export actual experiment charts for the portfolio README."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from quality_intelligence.data import FEATURES

root = Path(__file__).resolve().parents[1]
result = json.loads((root / "reports/results.json").read_text())
models = pd.DataFrame(result["models"]).set_index("model")
predictions = pd.read_parquet(root / "reports/predictions.parquet")
importance = pd.read_csv(root / "reports/importance.csv", index_col=0).iloc[:, 0]
optimization = pd.read_csv(root / "reports/optimization.csv")
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
models[["pr_auc", "recall", "precision"]].plot.bar(
    ax=axes[0, 0], rot=15, color=["#008f87", "#2574a9", "#ecaa42"]
)
axes[0, 0].set_title("Held-out model performance")
axes[0, 0].set_ylim(0, 1)
observed, risk = calibration_curve(
    predictions.defect, predictions[result["selected_model"]], n_bins=10, strategy="quantile"
)
axes[0, 1].plot(risk, observed, "o-", color="#008f87")
axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
axes[0, 1].set(
    xlabel="Predicted risk", ylabel="Observed defect fraction", title="Test reliability curve"
)
importance.sort_values().plot.barh(ax=axes[1, 0], color="#2574a9")
axes[1, 0].set_title("Mean absolute SHAP on held-out sample")
summary = optimization.groupby("method").oracle_probability.agg(["mean", "std"])
axes[1, 1].bar(
    summary.index, summary["mean"] * 100, yerr=summary["std"] * 100, color="#008f87", capsize=4
)
axes[1, 1].set(
    ylabel="Simulator defect probability (%)", title="Three search seeds (error bars: SD)"
)
fig.suptitle("Synthetic battery manufacturing • Actual experiment outputs", fontsize=16)
fig.savefig(root / "assets/evaluation.png", dpi=150)
plt.close(fig)
sample = pd.read_parquet(root / "reports/explanation_sample.parquet")
values = np.load(root / "reports/shap.npy")
fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
for i, feature in enumerate(importance.index):
    raw = sample[feature].to_numpy()
    ax.scatter(
        values[:, FEATURES.index(feature)],
        np.full(len(values), i),
        c=raw,
        cmap="coolwarm",
        s=12,
        alpha=0.6,
    )
ax.set(
    yticks=range(len(importance)),
    yticklabels=importance.index,
    xlabel="Raw model SHAP contribution",
    title="SHAP summary • Color represents feature value",
)
fig.savefig(root / "assets/shap_summary.png", dpi=150)

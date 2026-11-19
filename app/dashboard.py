"""Interactive views of actual persisted experiment outputs."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from quality_intelligence.analysis import explain
from quality_intelligence.data import CENTERS, FEATURES, SCALES
from quality_intelligence.modeling import metrics, predict

ROOT = Path(os.environ.get("QUALITY_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="Battery Quality Intelligence", page_icon="⚡", layout="wide")
st.title("Battery Manufacturing Quality Intelligence")
st.caption("Independent academic portfolio • Synthetic manufacturing demonstration")
st.warning(
    "Simulated data and operating ranges. Predictions and investigation hypotheses are not "
    "validated for physical batteries. Individual risks have no validated confidence interval."
)


@st.cache_data
def load_data():
    return (
        pd.read_parquet(ROOT / "data/modules.parquet"),
        pd.read_parquet(ROOT / "reports/predictions.parquet"),
        json.loads((ROOT / "reports/results.json").read_text()),
    )


@st.cache_resource
def load_system():
    return joblib.load(ROOT / "models/system.joblib")


if not (ROOT / "models/system.joblib").exists():
    st.info("No trained artifacts. Run: python scripts/run_pipeline.py --seed 42")
    st.stop()
try:
    with st.spinner("Loading evaluated manufacturing outputs…"):
        frame, predictions, results = load_data()
        system = load_system()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Artifacts could not be loaded. Re-run pipeline. Details: {error}")
    st.stop()

name = system["selected"]
tabs = st.tabs(
    [
        "Quality overview",
        "Model performance",
        "Anomaly detection",
        "Explainable AI & RCA",
        "Process optimization",
        "Interactive prediction",
    ]
)
with tabs[0]:
    for col, label, value in zip(
        st.columns(4),
        ["Modules", "Pass rate", "Defect rate", "First-pass yield"],
        [
            f"{len(frame):,}",
            f"{1 - frame.defect.mean():.2%}",
            f"{frame.defect.mean():.2%}",
            f"{1 - frame.defect.mean():.2%}",
        ],
    ):
        col.metric(label, value)
    st.caption("One inspection per module; no rework simulation, so FPY equals pass rate.")
    overview_left, overview_right = st.columns(2)
    overview_left.plotly_chart(
        px.bar(
            frame[frame.defect == 1].defect_category.value_counts().reset_index(),
            x="defect_category",
            y="count",
            title="Simulated defect categories",
        )
    )
    overview_right.plotly_chart(
        px.line(
            frame.groupby("batch", as_index=False).defect.mean(),
            x="batch",
            y="defect",
            title="Batch quality trends",
        )
    )
    st.plotly_chart(
        px.bar(frame.groupby("equipment", as_index=False).defect.mean(), x="equipment", y="defect")
    )
with tabs[1]:
    st.caption(
        f"Selected on tuning PR-AUC: {name}. All displayed comparisons use held-out test rows."
    )
    st.dataframe(pd.DataFrame(results["models"]).drop(columns="confusion_matrix"))
    threshold = st.slider(
        "Classification threshold", 0.0, 1.0, float(results["policies"]["balanced_cost"]), 0.005
    )
    selected_metrics = metrics(predictions.defect, predictions[name], threshold)
    for col, key in zip(st.columns(4), ["precision", "recall", "pr_auc", "f1"]):
        col.metric(key.replace("_", " ").title(), f"{selected_metrics[key]:.4f}")
    performance_left, performance_right = st.columns(2)
    performance_left.plotly_chart(
        px.imshow(
            selected_metrics["confusion_matrix"],
            text_auto=True,
            x=["Predicted pass", "Predicted defect"],
            y=["Actual pass", "Actual defect"],
            color_continuous_scale="Teal",
            title="Held-out confusion matrix",
        )
    )
    precision, recall, _ = precision_recall_curve(predictions.defect, predictions[name])
    performance_right.plotly_chart(
        px.line(
            x=recall,
            y=precision,
            title="Precision–recall curve",
            labels={"x": "Defect recall", "y": "Precision"},
        )
    )
    observed, predicted = calibration_curve(
        predictions.defect, predictions[name], n_bins=10, strategy="quantile"
    )
    fig = px.line(
        x=predicted, y=observed, markers=True, labels={"x": "Predicted risk", "y": "Observed rate"}
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration")
    st.plotly_chart(fig)
    st.dataframe(pd.read_csv(ROOT / "reports/calibration.csv"))
    st.json(results["policy_test_metrics"])
with tabs[2]:
    for col, key in zip(
        st.columns(4), ["precision", "recall", "false_positive_rate", "inspection_burden"]
    ):
        col.metric(key.replace("_", " ").title(), f"{results['anomaly'][key]:.2%}")
    st.caption(
        "Anomalies and defects differ. False-alarm rate is measured against synthetic defect labels."
    )
    anomaly_left, anomaly_right = st.columns(2)
    anomaly_left.plotly_chart(
        px.histogram(
            predictions,
            x="anomaly_score",
            color=predictions.defect.astype(str),
            title="Isolation Forest score distribution",
        )
    )
    for group, container in zip(["equipment", "batch"], [anomaly_right, st]):
        container.plotly_chart(
            px.bar(
                predictions.groupby(group, as_index=False).anomaly_alert.mean(),
                x=group,
                y="anomaly_alert",
                title=f"Anomaly alert rate by {group}",
            )
        )
    merged = frame.merge(predictions[["module_id", "anomaly_alert"]], on="module_id")
    st.dataframe(merged.groupby("anomaly_alert")[FEATURES].mean())
    st.dataframe(
        pd.crosstab(
            predictions.anomaly_alert, predictions[name] >= results["policies"]["balanced_cost"]
        )
    )
with tabs[3]:
    importance = pd.read_csv(ROOT / "reports/importance.csv", index_col=0)
    rca_left, rca_right = st.columns(2)
    rca_left.plotly_chart(px.bar(importance, orientation="h", title="Global mean absolute SHAP"))
    rca_right.subheader("Batch investigation priorities")
    rca_right.dataframe(pd.read_csv(ROOT / "reports/batches.csv").head(8), hide_index=True)
    rca_right.caption(
        "Ranked by predicted risk. Observed rates and SHAP associations support "
        "investigation hypotheses, not confirmed root causes."
    )
    sample = pd.read_parquet(ROOT / "reports/explanation_sample.parquet")
    values = np.load(ROOT / "reports/shap.npy")
    long = pd.DataFrame(values, columns=FEATURES).melt(var_name="feature", value_name="SHAP")
    st.plotly_chart(px.strip(long, x="SHAP", y="feature", title="SHAP summary distribution"))
    row = st.selectbox("Held-out module explanation", range(len(sample)))
    st.plotly_chart(px.bar(x=values[row], y=FEATURES, orientation="h"))
    st.caption(
        "SHAP explains the raw classifier score; calibration is a separate transformation. "
        "Importance magnitude units depend on the selected model family."
    )
    st.dataframe(pd.read_csv(ROOT / "reports/batches.csv"))
    report = (ROOT / "reports/investigation.md").read_text(encoding="utf-8")
    st.markdown(report)
    st.download_button("Download investigation", report, "investigation.md")
with tabs[4]:
    st.caption(
        "Search changes common control settings over fixed held-out contexts. "
        "Oracle probability is independent of the surrogate prediction."
    )
    optimization = pd.read_csv(ROOT / "reports/optimization.csv")
    summary = optimization.groupby("method").oracle_probability.mean()
    for col, method in zip(st.columns(3), ["baseline", "random", "constrained"]):
        col.metric(f"{method.title()} • simulator defect probability", f"{summary[method]:.3%}")
    st.plotly_chart(
        px.bar(
            optimization.groupby("method", as_index=False)[
                ["predicted_risk", "oracle_probability"]
            ].mean(),
            x="method",
            y=["predicted_risk", "oracle_probability"],
            barmode="group",
        )
    )
    st.subheader("Process settings and constraint validation")
    st.dataframe(optimization, hide_index=True)
    with st.expander("Full experiment report"):
        st.markdown((ROOT / "reports/evaluation.md").read_text(encoding="utf-8"))
with tabs[5]:
    user = {}
    inputs = st.columns(3)
    for i, col in enumerate(FEATURES):
        user[col] = inputs[i % 3].number_input(
            col,
            value=float(system["defaults"][col]),
            step=float(SCALES[col] / 10),
            key=f"input_{i}",
        )
    candidate = pd.DataFrame([user])
    p = float(predict(system["bundle"], candidate)[0])
    st.metric("Synthetic defect probability", f"{p:.2%}")
    st.write("Fail / inspect" if p >= results["policies"]["balanced_cost"] else "Pass prediction")
    st.write("Risk band: " + ("high" if p >= 0.2 else "moderate" if p >= 0.05 else "low"))
    if any(abs(user[k] - CENTERS[k]) > 3 * SCALES[k] for k in FEATURES):
        st.warning(
            "Inputs extend beyond three nominal simulation scales; extrapolation is unreliable."
        )
    contributions = explain(system["bundle"]["model"], candidate)[0]
    st.plotly_chart(
        px.bar(x=contributions, y=FEATURES, orientation="h", title="Raw score SHAP contributors")
    )

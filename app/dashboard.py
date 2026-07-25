"""
Phase 7 — Analyst-Facing Dashboard.

Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from explain import build_explanation, FEATURE_LABELS  # noqa: E402

st.set_page_config(page_title="Behavioral Anomaly Detection — SOC Dashboard",
                    layout="wide", page_icon="🛡️")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    with open("data/processed/detection_models.pkl", "rb") as f:
        det = pickle.load(f)
    with open("data/processed/classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    dev = pd.read_csv("data/processed/deviation_features.csv")
    raw = pd.read_csv("data/raw/access_logs.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    return det, clf, dev, raw


det, clf_bundle, dev, raw = load_artifacts()

feature_cols = det["feature_cols"]
xgb_model = det["xgboost"]
scaler = det["scaler"]
test_idx = det["test_idx"]
xgb_scores = det["xgb_scores"]

dev_dummies = pd.get_dummies(dev, columns=["entity_type"], prefix="etype")
test_df = dev_dummies.loc[test_idx].reset_index(drop=True).copy()
test_df["xgb_score"] = xgb_scores

explainer = shap.TreeExplainer(xgb_model)


# ---------------------------------------------------------------------------
# Sidebar — alert budget control
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ SOC Controls")
budget_pct = st.sidebar.slider("Alert budget (top X% of sessions)", 0.1, 10.0, 1.0, 0.1)
n_alerts = max(1, int(len(test_df) * budget_pct / 100))

sorted_df = test_df.sort_values("xgb_score", ascending=False).reset_index(drop=True)
alert_df = sorted_df.head(n_alerts).copy()
alert_df["is_true_anomaly"] = alert_df["label"] != "normal"

precision = alert_df["is_true_anomaly"].mean()
recall = alert_df["is_true_anomaly"].sum() / max((test_df["label"] != "normal").sum(), 1)
fp_rate = 1 - precision

st.sidebar.metric("Alerts at this budget", n_alerts)
st.sidebar.metric("Precision", f"{precision:.1%}")
st.sidebar.metric("Recall", f"{recall:.1%}")
st.sidebar.metric("False positive rate", f"{fp_rate:.1%}")
st.sidebar.caption("Directly demos the hackathon's stated evaluation criterion: "
                    "false-positive rate at a realistic analyst alert budget.")

# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------

st.title("AI-Powered Behavioral Anomaly Detection")
st.caption("Sequence-aware intrusion detection · Explainable risk scoring · SOC analyst view")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total sessions (test window)", f"{len(test_df):,}")
k2.metric("True anomalies (test window)", int((test_df['label'] != 'normal').sum()))
k3.metric("Current alert queue", n_alerts)
k4.metric("Precision @ budget", f"{precision:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# 7.1 — Ranked alert queue with predicted attack type + explanation
# ---------------------------------------------------------------------------

st.subheader("Ranked Alert Queue")

X_alerts = alert_df[feature_cols].fillna(0).values
X_alerts_s = scaler.transform(X_alerts)

# predicted attack type via Phase 5 classifier
clf_model, clf_scaler, le, clf_feats = (
    clf_bundle["model"], clf_bundle["scaler"], clf_bundle["label_encoder"], clf_bundle["feature_cols"]
)
X_clf = alert_df[clf_feats].fillna(0).values
X_clf_s = clf_scaler.transform(X_clf)
pred_type = le.inverse_transform(clf_model.predict(X_clf_s))

shap_vals = explainer.shap_values(X_alerts_s)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

explanations = [
    build_explanation(shap_vals[i], feature_cols, X_alerts[i])
    for i in range(len(alert_df))
]

display_df = pd.DataFrame({
    "entity_id": alert_df["entity_id"].values,
    "predicted_type": pred_type,
    "true_label": alert_df["label"].values,
    "risk_score": alert_df["xgb_score"].round(3).values,
    "explanation": explanations,
    "session_id": alert_df["session_id"].values,
})

st.dataframe(
    display_df[["entity_id", "predicted_type", "true_label", "risk_score", "explanation"]],
    use_container_width=True, height=350,
)

# ---------------------------------------------------------------------------
# 7.2 / 7.3 — Row detail: entity history + SHAP mini-chart
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Alert Detail")

selected_session = st.selectbox(
    "Select an alert to inspect",
    options=display_df["session_id"].tolist(),
    format_func=lambda sid: f"{sid} — {display_df[display_df.session_id==sid]['entity_id'].values[0]} "
                             f"({display_df[display_df.session_id==sid]['predicted_type'].values[0]})",
)

sel_row_idx = display_df[display_df.session_id == selected_session].index[0]
sel_entity = display_df.loc[sel_row_idx, "entity_id"]
sel_ts = raw.loc[raw.session_id == selected_session, "timestamp"].values[0]

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown(f"**Entity history — `{sel_entity}`**")
    history = raw[raw.entity_id == sel_entity].sort_values("timestamp").copy()
    history["is_flagged"] = history["session_id"] == selected_session

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["timestamp"], y=history["resource_accessed"],
        mode="markers", marker=dict(size=6, color="#4f81bd"),
        name="session", hovertext=history["session_id"],
    ))
    flagged_pt = history[history.is_flagged]
    fig.add_trace(go.Scatter(
        x=flagged_pt["timestamp"], y=flagged_pt["resource_accessed"],
        mode="markers", marker=dict(size=16, color="#c0504d", symbol="x"),
        name="flagged session",
    ))
    fig.update_layout(height=380, template="plotly_white",
                       xaxis_title="time", yaxis_title="resource accessed",
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Top contributing features**")
    row_pos = alert_df.index.get_loc(alert_df[alert_df.session_id == selected_session].index[0])
    row_shap = shap_vals[row_pos]
    row_X = X_alerts[row_pos]
    order = np.argsort(-np.abs(row_shap))[:6]
    chart_df = pd.DataFrame({
        "feature": [feature_cols[i] for i in order],
        "shap_value": [row_shap[i] for i in order],
    }).sort_values("shap_value")
    fig2 = px.bar(chart_df, x="shap_value", y="feature", orientation="h",
                  color="shap_value", color_continuous_scale=["#4f81bd", "#c0504d"])
    fig2.update_layout(height=380, template="plotly_white", showlegend=False,
                        coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.info(display_df.loc[sel_row_idx, "explanation"])

st.divider()
st.caption(
    "MVP scope: statistical/tree-based pipeline (Isolation Forest + SMOTE-XGBoost + "
    "per-entity Markov transition features), chosen over deep sequence models to fit "
    "the hackathon timeline. See report for full methodology and known limitations."
)

"""
Phase 7 — Analyst-Facing Dashboard.
Run: streamlit run app/dashboard_gemini.py
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
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from explain import build_explanation, FEATURE_LABELS  # noqa: E402

st.set_page_config(page_title="Sentinel: Threat Triage Console",
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
# Sidebar — Command Center Filters
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ Sentinel Controls")
st.sidebar.caption("Global Triage Parameters")

# Added Time Window (Mocked for dashboard realism)
st.sidebar.date_input(
    "Analysis Window", 
    value=(datetime.date(2026, 7, 24), datetime.date(2026, 7, 25))
)

# Budget Slider
budget_pct = st.sidebar.slider("Alert Budget (Top X% of events)", 0.1, 10.0, 1.0, 0.1)
n_alerts = max(1, int(len(test_df) * budget_pct / 100))

# Threat Type Filters (Mocked for UI flow, applied to display_df later)
threat_filter = st.sidebar.multiselect(
    "Filter Threat Types",
    options=["brute_force", "credential_stuffing", "lateral_movement", "exfiltration", "impossible_travel", "device_spoofing", "insider_drift"],
    default=["brute_force", "credential_stuffing", "lateral_movement", "exfiltration", "impossible_travel", "device_spoofing", "insider_drift"]
)

st.sidebar.divider()
st.sidebar.caption("Directly demos the hackathon's stated evaluation criterion: "
                    "false-positive rate at a realistic analyst alert budget.")

# ---------------------------------------------------------------------------
# Data Processing for Queue
# ---------------------------------------------------------------------------
sorted_df = test_df.sort_values("xgb_score", ascending=False).reset_index(drop=True)
alert_df = sorted_df.head(n_alerts).copy()
alert_df["is_true_anomaly"] = alert_df["label"] != "normal"

precision = alert_df["is_true_anomaly"].mean()
recall = alert_df["is_true_anomaly"].sum() / max((test_df["label"] != "normal").sum(), 1)
fp_rate = 1 - precision

# predicted attack type via Phase 5 classifier
clf_model, clf_scaler, le, clf_feats = (
    clf_bundle["model"], clf_bundle["scaler"], clf_bundle["label_encoder"], clf_bundle["feature_cols"]
)
X_alerts = alert_df[feature_cols].fillna(0).values
X_alerts_s = scaler.transform(X_alerts)

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
    "session_id": alert_df["session_id"].values,
    "entity_id": alert_df["entity_id"].values,
    "predicted_type": pred_type,
    "true_label": alert_df["label"].values,
    "risk_score": alert_df["xgb_score"].round(3).values,
    "explanation": explanations,
})

# Apply threat type filter from sidebar
display_df = display_df[display_df["predicted_type"].isin(threat_filter)]

# ---------------------------------------------------------------------------
# Header KPIs (Moved out of sidebar for prominence)
# ---------------------------------------------------------------------------

st.title("🛡️ Sentinel: Threat Triage Console")
st.caption("Sequence-Aware Intrusion Detection & Explainable AI Pipeline")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Active Critical Alerts", len(display_df))
k2.metric("Precision @ Budget", f"{precision:.1%}")
k3.metric("Recall @ Budget", f"{recall:.1%}")
k4.metric("False Positive Rate", f"{fp_rate:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# 7.1 — Ranked Alert Queue (With Risk Color Coding)
# ---------------------------------------------------------------------------

st.subheader("Ranked Alert Queue")

# Color formatting for the risk score
def style_risk_score(val):
    if val >= 0.9:
        return 'color: #ff4b4b; font-weight: bold' # Red
    elif val >= 0.7:
        return 'color: #ffa421; font-weight: bold' # Orange/Yellow
    return 'color: #2e8b57' # Green

styled_df = display_df[["entity_id", "predicted_type", "risk_score", "explanation"]].style.map(
    style_risk_score, subset=['risk_score']
)

st.dataframe(styled_df, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# 7.2 / 7.3 — Progressive Disclosure Drill-Down
# ---------------------------------------------------------------------------

st.divider()

# Wrapped in an expander for cleaner UI
with st.expander("🔍 Deep Dive: Threat Intelligence", expanded=True):
    
    selected_session = st.selectbox(
        "Select an alert to inspect context:",
        options=display_df["session_id"].tolist(),
        format_func=lambda sid: f"{sid} — {display_df[display_df.session_id==sid]['entity_id'].values[0]} "
                                 f"({display_df[display_df.session_id==sid]['predicted_type'].values[0]})",
    )

    if selected_session:
        sel_row_idx = display_df[display_df.session_id == selected_session].index[0]
        sel_entity = display_df.loc[sel_row_idx, "entity_id"]
        sel_ts = raw.loc[raw.session_id == selected_session, "timestamp"].values[0]

        col_a, col_b = st.columns([2, 1])

        # Graph Redesign: Baseline Operating Hours vs Anomaly
        with col_a:
            st.markdown(f"**Entity Baseline vs Anomaly: `{sel_entity}`**")
            
            # Get user history and extract hour for baseline distribution
            history = raw[raw.entity_id == sel_entity].copy()
            history['hour'] = history['timestamp'].dt.hour
            
            # Count frequency of logins per hour to establish "Normal"
            baseline = history['hour'].value_counts().reset_index()
            baseline.columns = ['Hour of Day', 'Access Count']
            baseline = baseline.sort_values('Hour of Day')

            flagged_hour = pd.to_datetime(sel_ts).hour

            # Plot Baseline Bar Chart
            fig = px.bar(
                baseline, x='Hour of Day', y='Access Count', 
                template="plotly_white", opacity=0.7,
                color_discrete_sequence=['#4f81bd']
            )
            
            # Overlay Anomaly (Red dashed line)
            fig.add_vline(
                x=flagged_hour, line_dash="dash", line_color="#c0504d", line_width=3,
                annotation_text="🚨 Flagged Session", annotation_position="top right"
            )
            
            fig.update_layout(
                height=350, 
                xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-1, 24]),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Top Contributing Vectors (SHAP)**")
            
            # Locate the correct row in the original alert_df to map SHAP values
            row_pos = alert_df.index.get_loc(alert_df[alert_df.session_id == selected_session].index[0])
            row_shap = shap_vals[row_pos]
            row_X = X_alerts[row_pos]
            
            # Get top 5 features
            order = np.argsort(-np.abs(row_shap))[:5]
            chart_df = pd.DataFrame({
                "feature": [feature_cols[i] for i in order],
                "shap_value": [row_shap[i] for i in order],
            }).sort_values("shap_value")
            
            fig2 = px.bar(
                chart_df, x="shap_value", y="feature", orientation="h",
                color="shap_value", color_continuous_scale=["#4f81bd", "#c0504d"]
            )
            fig2.update_layout(
                height=350, template="plotly_white", showlegend=False,
                coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Highlight the generated NL explanation
        st.error(f"**Automated Triage Note:** {display_df.loc[sel_row_idx, 'explanation']}")

st.divider()
st.caption(
    "MVP scope: statistical/tree-based pipeline (Isolation Forest + SMOTE-XGBoost + "
    "per-entity Markov transition features), chosen over deep sequence models to fit "
    "the hackathon timeline. See report for full methodology and known limitations."
)
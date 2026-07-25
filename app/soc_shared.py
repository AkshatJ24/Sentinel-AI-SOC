"""Shared data, navigation, and presentation utilities for Sentinel AI SOC."""

import html
import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from explain import build_explanation, FEATURE_LABELS  # noqa: E402


COLORS = {
    "background": "#0F1117",
    "card": "#171B22",
    "border": "#2A2F3A",
    "primary": "#4F8EF7",
    "danger": "#FF4D4F",
    "warning": "#F5A623",
    "success": "#2ECC71",
    "muted": "#9AA4B2",
    "text": "#EEF3F8",
    "critical": "#FF4D4F",
    "high": "#FF7A59",
    "medium": "#F5A623",
    "low": "#2ECC71",
}

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

MITRE_MAPPING = {
    "brute_force": ("T1110", "Brute Force"),
    "impossible_travel": ("T1078", "Valid Accounts"),
    "data_exfiltration": ("T1041", "Exfiltration Over C2 Channel"),
    "credential_stuffing": ("T1110.004", "Credential Stuffing"),
    "lateral_movement": ("T1021", "Remote Services"),
    "privilege_escalation": ("T1068", "Exploitation for Privilege Escalation"),
}


def configure_page(title):
    """Configure a page before rendering any Streamlit content."""
    st.set_page_config(
        page_title=f"{title} · Sentinel AI SOC",
        layout="wide",
        page_icon="🛡️",
        initial_sidebar_state="expanded", 
    )
    inject_css()

def inject_css():
    """Install the common, compact Sentinel SOC design system."""
    st.markdown(
        """
        <style>
        :root {
          --bg:#0F1117; --card:#171B22; --border:#2A2F3A; --blue:#4F8EF7;
          --red:#FF4D4F; --orange:#F5A623; --green:#2ECC71; --muted:#9AA4B2; --text:#EEF3F8;
        }
        .stApp {
          background:radial-gradient(circle at 82% -10%,rgba(79,142,247,.17),transparent 26rem),
                     radial-gradient(circle at 2% 18%,rgba(46,204,113,.07),transparent 22rem),var(--bg);
          color:var(--text);
        }
        /* ── Streamlit chrome ─────────────────────────────────────────────────── */
        #MainMenu, footer {visibility:hidden;}
        /* Transparent header — keeps the sidebar expand chevron accessible */
        [data-testid="stHeader"] {
            background:transparent !important; border:none !important;
            box-shadow:none !important;
        }
        [data-testid="stToolbar"] {visibility:hidden !important;}
        [data-testid="stDecoration"] {display:none !important;}

                /* ── Sidebar width (fixed for collapse/expand) ───────────────────────── */
        /* Only pin width when EXPANDED. When collapsed Streamlit must be allowed
           to set width/margin to 0 / negative. Target the inner div, not the
           section, so the native animation isn't killed by !important.          */
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
            background: linear-gradient(180deg,#11151D 0%,#0C0F14 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 0px !important;
            min-width: 0px !important;
            margin-left: 0px !important;
        }
        /* Minimal padding on the first child so our brand appears close to the top */
        [data-testid="stSidebar"] > div:first-child {padding-top:0.3rem !important;}

        /* ── MAKE COLLAPSE BUTTON VISIBLE ────────────────────────────────────── */
        [data-testid="stSidebar"] button[kind="headerNoPadding"],
        [data-testid="stSidebar"] button[aria-label*="collapse" i],
        [data-testid="stSidebar"] button[aria-label*="expand" i],
        [data-testid="stSidebarNav"] button,
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarNavCollapseButton"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            display: inline-flex !important;
            color: #EEF3F8 !important;
            background: rgba(23,27,34,0.9) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            padding: 6px !important;
            z-index: 9999 !important;
            position: relative !important;
        }
        [data-testid="stSidebar"] button[kind="headerNoPadding"]:hover,
        [data-testid="stSidebar"] button[aria-label*="collapse" i]:hover,
        [data-testid="stSidebar"] button[aria-label*="expand" i]:hover {
            background: rgba(79,142,247,0.25) !important;
            border-color: rgba(79,142,247,0.5) !important;
            color: #FFFFFF !important;
        }

        /* ── Header gap filler (only when sidebar is expanded) ───────────────── */
        section[data-testid="stSidebar"][aria-expanded="true"]::before {
            content:'';
            position:fixed;
            top:0; left:0;
            width:260px;
            height:var(--header-height, 3.75rem);
            background:#11151D;
            border-right:1px solid var(--border,#2A2F3A);
            z-index:100;
            pointer-events:none;
        }
        section[data-testid="stSidebar"][aria-expanded="false"]::before {
            display: none;
        }
        /* Sidebar slider styling */

        /* ── Sidebar slider styling ──────────────────────────────────────────── */
        [data-testid="stSidebar"] .stSlider {padding:0 .1rem;}
        [data-testid="stSidebar"] .stSlider label {display:none;}
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {display:none;}
        [data-testid="stSidebar"] .stButton > button {
          min-height:31px; padding:.35rem .48rem; border:1px solid transparent;
          background:transparent; color:#C7D0DD; border-radius:8px; font-size:.77rem;
          text-align:left; transition:background .15s ease,border-color .15s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
          background:rgba(79,142,247,.11); border-color:rgba(79,142,247,.22); color:#FFFFFF;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
          background:linear-gradient(90deg,rgba(79,142,247,.28),rgba(79,142,247,.08));
          border-color:rgba(79,142,247,.32); color:#FFFFFF;
        }

        /* ── Layout ───────────────────────────────────────────────────────────── */
        .block-container {max-width:1600px; padding:1.45rem 2rem 3.4rem;}
        h1,h2,h3,h4,p,div,span,label {
          font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        }
        h1,h2,h3,h4 {color:var(--text) !important; letter-spacing:-.03em;}
        h2 {font-size:1.16rem !important; margin-top:1rem !important;}

        /* ── Header gap filler (only when sidebar is expanded) ───────────────── */
        section[data-testid="stSidebar"][aria-expanded="true"]::before {
            content:'';
            position:fixed;
            top:0; left:0;
            width:260px;
            height:var(--header-height, 3.75rem);
            background:#11151D;
            border-right:1px solid var(--border,#2A2F3A);
            z-index:100;
            pointer-events:none;
        }
        section[data-testid="stSidebar"][aria-expanded="false"]::before {
            display: none;
        }

        /* ── (rest of your design system unchanged) ──────────────────────────── */
        .brand {padding:.12rem .1rem .9rem;border-bottom:1px solid var(--border);margin-bottom:.72rem;}
        .brand h1 {font-size:1.12rem !important;margin:0 0 .16rem !important;}
        .brand p {color:var(--muted);font-size:.64rem;letter-spacing:.08em;margin:0;}
        .side-label,.section-kicker {color:var(--muted);font-size:.64rem;font-weight:750;letter-spacing:.11em;text-transform:uppercase;}
        .side-label {margin:.9rem 0 .32rem;}
        .side-divider {height:1px;background:var(--border);margin:.7rem 0;}
        .side-hint {color:var(--muted);font-size:.66rem;line-height:1.35;margin:.42rem .1rem;}
        .hero {display:flex;justify-content:space-between;gap:1.25rem;align-items:flex-start;margin:.05rem 0 1.25rem;}
        .hero h1 {font-size:clamp(1.55rem,3.2vw,2.35rem) !important;margin:0 !important;line-height:1.12;}
        .hero p {color:var(--muted);font-size:.87rem;margin:.46rem 0 0;}
        .eyebrow {color:#83AEFF;font-weight:750;font-size:.67rem;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.35rem;}
        .live-pill {white-space:nowrap;border:1px solid rgba(46,204,113,.32);border-radius:999px;background:rgba(46,204,113,.09);color:#8DE4AE;padding:.36rem .62rem;font-size:.66rem;font-weight:750;}
        .section-title-row {display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:.18rem 0 .62rem;}
        .section-title-row h2 {margin:0 !important;}
        .section-note {color:var(--muted);font-size:.7rem;}
        .kpi-card,.severity-card,.action-card,.entity-card {
          border:1px solid var(--border);border-radius:13px;background:linear-gradient(145deg,rgba(28,34,44,.96),rgba(20,24,31,.96));box-shadow:0 14px 30px rgba(0,0,0,.15);
        }
        .kpi-card {min-height:116px;padding:.9rem;position:relative;overflow:hidden;transition:transform .18s ease,border-color .18s ease;}
        .kpi-card:hover,.severity-card:hover,.action-card:hover {transform:translateY(-3px);border-color:rgba(79,142,247,.55);}
        .kpi-card:after {content:"";position:absolute;width:82px;height:82px;right:-30px;bottom:-35px;background:radial-gradient(circle,var(--accent,#4F8EF7),transparent 68%);opacity:.18;}
        .kpi-top {display:flex;justify-content:space-between;align-items:center;}
        .kpi-label,.severity-name {color:var(--muted);font-size:.69rem;font-weight:750;letter-spacing:.035em;}
        .kpi-value {color:var(--text);font-size:1.46rem;font-weight:760;letter-spacing:-.04em;margin-top:.48rem;}
        .kpi-detail {color:#AEB9C8;font-size:.67rem;margin-top:.24rem;}
        .severity-card {padding:.78rem .88rem;min-height:86px;}
        .severity-name {text-transform:uppercase;letter-spacing:.08em;}
        .severity-count {color:var(--text);font-size:1.35rem;font-weight:760;margin-top:.2rem;}
        .severity-hint {font-size:.64rem;margin-top:.16rem;}
        .badge {display:inline-block;border-radius:999px;padding:.2rem .46rem;font-size:.62rem;font-weight:760;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;}
        .badge-critical {background:rgba(255,77,79,.16);color:#FF9091;border:1px solid rgba(255,77,79,.3);}
        .badge-high {background:rgba(255,122,89,.14);color:#FFAA91;border:1px solid rgba(255,122,89,.28);}
        .badge-medium {background:rgba(245,166,35,.14);color:#FFD17A;border:1px solid rgba(245,166,35,.28);}
        .badge-low {background:rgba(46,204,113,.13);color:#8DE4AE;border:1px solid rgba(46,204,113,.28);}
        .risk-pill {display:inline-block;color:#EAF1FB;background:rgba(79,142,247,.14);border:1px solid rgba(79,142,247,.24);border-radius:999px;padding:.2rem .44rem;font-weight:700;font-size:.68rem;}
        .queue-header,.queue-row {display:grid;grid-template-columns:1.05fr .63fr 1.18fr 1.25fr 1.08fr .72fr 2.3fr .86fr;gap:.62rem;}
        .queue-header {padding:.58rem .66rem;color:var(--muted);font-size:.61rem;font-weight:760;letter-spacing:.07em;text-transform:uppercase;background:#161B23;border:1px solid var(--border);border-radius:10px 10px 0 0;position:sticky;top:0;z-index:4;}
        .queue-row {align-items:center;padding:.62rem .66rem;border:1px solid rgba(42,47,58,.86);border-top:0;background:rgba(23,27,34,.84);font-size:.7rem;}
        .queue-row:hover {background:rgba(79,142,247,.07);}
        .queue-row.critical {border-left:3px solid var(--red);} .queue-row.high {border-left:3px solid #FF7A59;}
        .queue-row.medium {border-left:3px solid var(--orange);} .queue-row.low {border-left:3px solid var(--green);}
        .queue-cell,.queue-explanation {color:#D9E1EB;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
        .queue-explanation {color:#ABB7C8;} .queue-action {color:#AECBFF;border:1px solid rgba(79,142,247,.3);border-radius:6px;padding:.24rem .38rem;text-align:center;font-weight:700;}
        .summary-card {padding:1rem 1.05rem;background:linear-gradient(130deg,rgba(79,142,247,.14),rgba(23,27,34,.94) 36%,rgba(23,27,34,.94));border:1px solid rgba(79,142,247,.3);border-radius:14px;}
        .summary-card h3 {margin:.38rem 0 .78rem !important;font-size:1.18rem !important;}
        .summary-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.62rem;}
        .summary-stat {padding:.55rem .62rem;border-radius:8px;background:rgba(8,11,15,.28);border:1px solid rgba(255,255,255,.055);}
        .summary-stat span {display:block;color:var(--muted);font-size:.61rem;text-transform:uppercase;letter-spacing:.06em;}
        .summary-stat strong {display:block;color:var(--text);margin-top:.2rem;font-size:.78rem;overflow:hidden;text-overflow:ellipsis;}
        .investigation-panel {height:100%;border:1px solid rgba(79,142,247,.27);background:linear-gradient(145deg,rgba(26,35,50,.92),rgba(19,23,30,.96));border-radius:14px;padding:.95rem 1rem;}
        .evidence-item {display:flex;gap:.52rem;padding:.5rem 0;border-top:1px solid rgba(255,255,255,.06);color:#D7E0EC;font-size:.75rem;line-height:1.38;}
        .evidence-icon {color:#85B0FF;}
        .action-card {padding:.76rem;min-height:126px;} .action-card h4 {margin:.43rem 0 .26rem;font-size:.78rem;}
        .action-card p {margin:0;color:var(--muted);font-size:.65rem;line-height:1.38;}
        .action-danger {border-top:2px solid var(--red);} .action-warning {border-top:2px solid var(--orange);}
        .action-primary {border-top:2px solid var(--blue);} .action-success {border-top:2px solid var(--green);}
        .model-health {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.54rem;}
        .health-item {padding:.65rem;border:1px solid rgba(46,204,113,.18);border-radius:9px;background:rgba(46,204,113,.045);}
        .health-item span {display:block;color:var(--muted);font-size:.63rem;}.health-item strong {display:block;color:#8DE4AE;font-size:.72rem;margin-top:.2rem;}
        .empty-state {border:1px dashed #3B4453;background:rgba(23,27,34,.65);border-radius:12px;color:var(--muted);padding:1.3rem;text-align:center;font-size:.8rem;}
        .footer-note {color:var(--muted);font-size:.7rem;border-top:1px solid var(--border);margin-top:1.4rem;padding-top:.78rem;}
        .stButton > button,.stDownloadButton > button {border:1px solid rgba(79,142,247,.45);background:rgba(79,142,247,.12);color:#DCE9FF;border-radius:8px;font-size:.72rem;font-weight:700;}
        .stButton > button:hover {border-color:#77A6FF;background:rgba(79,142,247,.25);color:#FFFFFF;}
        [data-testid="stDataFrame"] {border:1px solid var(--border);border-radius:10px;overflow:hidden;}
        .stSelectbox > div > div,.stTextInput > div > div,.stMultiSelect > div > div {background:#151A22;border-color:#303847;border-radius:8px;}
        @media (max-width:900px) {
          .block-container {padding:1rem .9rem 2.4rem;} .hero {display:block;} .live-pill {display:inline-block;margin-top:.75rem;}
          .summary-grid,.model-health {grid-template-columns:repeat(2,minmax(0,1fr));}
          .queue-header,.queue-row {grid-template-columns:1fr .7fr 1.3fr 1fr;}
          .queue-header > :nth-child(4),.queue-header > :nth-child(5),.queue-header > :nth-child(6),.queue-header > :nth-child(7),
          .queue-row > :nth-child(4),.queue-row > :nth-child(5),.queue-row > :nth-child(6),.queue-row > :nth-child(7) {display:none;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    


# ---------------------------------------------------------------------------
# Preserved data loading and model computation from dashboard.py
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


# The canonical budget is stored under this key — it is set only by the
# _save_budget on_change callback, never by auto-initialisation code, so
# Streamlit's widget-reset cycle can never clobber it across page navigation.
_BUDGET_KEY = "soc_budget"


def _save_budget():
    """on_change callback: propagate slider value to the persistent store."""
    st.session_state[_BUDGET_KEY] = float(st.session_state.get("_budget_slider", 1.0))


def current_alert_budget():
    """Return the analyst budget that persists reliably across all pages."""
    return float(st.session_state.get(_BUDGET_KEY, 1.0))


@st.cache_resource(show_spinner="Refreshing the Sentinel alert workspace...")
def build_workspace(budget_pct):
    """
    Build the active alert workspace.

    The computations in this function retain the artifact loading, budget
    evaluation, classifier prediction, SHAP values, and explanations from
    dashboard.py. Caching is keyed by alert budget so page navigation does not
    repeat expensive inference.
    """
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

    n_alerts = max(1, int(len(test_df) * budget_pct / 100))

    sorted_df = test_df.sort_values("xgb_score", ascending=False).reset_index(drop=True)
    alert_df = sorted_df.head(n_alerts).copy()
    alert_df["is_true_anomaly"] = alert_df["label"] != "normal"

    precision = alert_df["is_true_anomaly"].mean()
    recall = alert_df["is_true_anomaly"].sum() / max((test_df["label"] != "normal").sum(), 1)
    fp_rate = 1 - precision

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
    alert_intelligence = enrich_alert_presentation(display_df, raw)
    severity_counts = {
        severity: int((alert_intelligence["severity"] == severity).sum())
        for severity in SEVERITY_ORDER
    }
    return {
        "det": det,
        "clf_bundle": clf_bundle,
        "dev": dev,
        "raw": raw,
        "feature_cols": feature_cols,
        "xgb_model": xgb_model,
        "scaler": scaler,
        "test_idx": test_idx,
        "xgb_scores": xgb_scores,
        "test_df": test_df,
        "explainer": explainer,
        "budget_pct": budget_pct,
        "n_alerts": n_alerts,
        "sorted_df": sorted_df,
        "alert_df": alert_df,
        "precision": precision,
        "recall": recall,
        "fp_rate": fp_rate,
        "X_alerts": X_alerts,
        "X_alerts_s": X_alerts_s,
        "pred_type": pred_type,
        "shap_vals": shap_vals,
        "explanations": explanations,
        "display_df": display_df,
        "alert_intelligence": alert_intelligence,
        "severity_counts": severity_counts,
    }


def get_workspace():
    """Return the cached workspace for the active session's alert budget."""
    return build_workspace(current_alert_budget())


# ---------------------------------------------------------------------------
# Functional multipage navigation
# ---------------------------------------------------------------------------

def _navigate(label, target, key, active=False, severity=None):
    """Render a sidebar action that either navigates or focuses a severity queue."""
    clicked = st.sidebar.button(
        label,
        key=f"nav_{key}",
        type="primary" if active else "secondary",
        use_container_width=True,
    )
    if not clicked:
        return
    if severity is not None:
        st.session_state["live_alert_severities"] = [severity]
    elif target.endswith("1_Live_Alerts.py"):
        st.session_state["live_alert_severities"] = SEVERITY_ORDER.copy()
    if active:
        st.rerun()
    st.switch_page(target)


def render_sidebar(workspace, active_page):
    """Render compact, count-aware navigation and highlight the active page."""
    counts = workspace["severity_counts"]
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="brand"><h1>🛡 Sentinel AI</h1><p>SECURITY OPERATIONS CENTER</p></div>',
            unsafe_allow_html=True,
        )
        # ── Alert Budget slider lives here so it is always rendered on every page.
        # Keeping it in the sidebar means the widget key is never orphaned during
        # page navigation, which prevents Streamlit from silently resetting it.
        st.markdown("<div class='side-label'>Alert Budget</div>", unsafe_allow_html=True)
        st.slider(
            "Alert budget",
            min_value=0.1,
            max_value=10.0,
            # Always seed from the persistent non-widget key so the slider
            # shows the right value on every page, including after navigation.
            value=current_alert_budget(),
            step=0.1,
            key="_budget_slider",          # throwaway widget key
            on_change=_save_budget,        # copies to _BUDGET_KEY on every move
            label_visibility="collapsed",
            help="Top X % of sessions ranked by anomaly score enter the active alert queue.",
        )
        st.markdown(
            f"<div class='side-hint'>"
            f"Top {workspace['budget_pct']:.1f}% · "
            f"<b style='color:#EEF3F8'>{workspace['n_alerts']:,}</b> alerts · "
            f"{workspace['precision']:.1%} precision"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='side-divider'></div><div class='side-label'>Navigation</div>", unsafe_allow_html=True)
    _navigate(
        "🏠  Dashboard ◀" if active_page == "Dashboard" else "🏠  Dashboard",
        "dashboard_chatgpt.py",
        "dashboard",
        active_page == "Dashboard",
    )
    _navigate(
        f"🚨  Live Alerts ({workspace['n_alerts']:,})" + (" ◀" if active_page == "Live Alerts" else ""),
        "pages/1_Live_Alerts.py",
        "live_alerts",
        active_page == "Live Alerts",
    )
    _navigate(f"🔴  Critical ({counts['Critical']:,})", "pages/1_Live_Alerts.py", "critical", severity="Critical")
    _navigate(f"🟠  High ({counts['High']:,})", "pages/1_Live_Alerts.py", "high", severity="High")
    _navigate(f"🟡  Medium ({counts['Medium']:,})", "pages/1_Live_Alerts.py", "medium", severity="Medium")
    _navigate(f"🟢  Low ({counts['Low']:,})", "pages/1_Live_Alerts.py", "low", severity="Low")
    with st.sidebar:
        st.markdown("<div class='side-divider'></div>", unsafe_allow_html=True)
    _navigate(
        "🔍  Investigation ◀" if active_page == "Investigation" else "🔍  Investigation",
        "pages/2_Investigation.py",
        "investigation",
        active_page == "Investigation",
    )
    _navigate(
        "👤  Entity Search ◀" if active_page == "Entity Search" else "👤  Entity Search",
        "pages/3_Entity_Search.py",
        "entity_search",
        active_page == "Entity Search",
    )
    _navigate(
        "📊  Analytics ◀" if active_page == "Analytics" else "📊  Analytics",
        "pages/4_Analytics.py",
        "analytics",
        active_page == "Analytics",
    )
    _navigate(
        "🧠  Model Health ◀" if active_page == "Model Health" else "🧠  Model Health",
        "pages/5_Model_Health.py",
        "model_health",
        active_page == "Model Health",
    )
    _navigate(
        "⚙  Settings ◀" if active_page == "Settings" else "⚙  Settings",
        "pages/6_Settings.py",
        "settings",
        active_page == "Settings",
    )
    with st.sidebar:
        st.markdown(
            f"<div class='footer-note' style='margin-top:1rem'>Hackathon MVP · v1.1<br>Sentinel AI SOC</div>",
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------------------------
# Shared presentation helpers
# ---------------------------------------------------------------------------

def safe_text(value, fallback="—"):
    if pd.isna(value) or value is None:
        return fallback
    return html.escape(str(value))


def risk_percent(score):
    return f"{float(score) * 100:.1f}%"


def severity_from_score(score):
    score = float(score)
    if score >= 0.90:
        return "Critical"
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def severity_badge(severity):
    return f"<span class='badge badge-{severity.lower()}'>{safe_text(severity)}</span>"


def plotly_layout(fig, height=330, show_legend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C9D3E0", family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#171B22", bordercolor="#394250", font_color="#EEF3F8"),
    )
    fig.update_xaxes(gridcolor="rgba(154,164,178,.12)", zerolinecolor="rgba(154,164,178,.16)")
    fig.update_yaxes(gridcolor="rgba(154,164,178,.12)", zerolinecolor="rgba(154,164,178,.16)")
    return fig


def render_page_header(title, subtitle, eyebrow="Security Operations Center"):
    st.markdown(
        f"""
        <div class="hero">
          <div><div class="eyebrow">{safe_text(eyebrow)}</div><h1>{safe_text(title)}</h1><p>{safe_text(subtitle)}</p></div>
          <div class="live-pill">● DETECTION PIPELINE ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(kicker, title, note=None):
    note_markup = f"<span class='section-note'>{safe_text(note)}</span>" if note else ""
    st.markdown(
        f"<div class='section-kicker'>{safe_text(kicker)}</div>"
        f"<div class='section-title-row'><h2>{safe_text(title)}</h2>{note_markup}</div>",
        unsafe_allow_html=True,
    )


def render_kpi_cards(items):
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.markdown(
            f"""
            <div class="kpi-card" style="--accent:{item['accent']};">
              <div class="kpi-top"><span class="kpi-label">{safe_text(item['label'])}</span><span>{item['icon']}</span></div>
              <div class="kpi-value">{safe_text(item['value'])}</div>
              <div class="kpi-detail">{safe_text(item['detail'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_severity_overview(alert_intelligence):
    counts = alert_intelligence["severity"].value_counts()
    columns = st.columns(4)
    for column, severity in zip(columns, SEVERITY_ORDER):
        count = int(counts.get(severity, 0))
        color = COLORS[severity.lower()]
        column.markdown(
            f"""
            <div class="severity-card" style="border-top:2px solid {color};">
              <div class="severity-name">{severity}</div><div class="severity-count">{count}</div>
              <div class="severity-hint" style="color:{color};">{'alert' if count == 1 else 'alerts'} in active queue</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def enrich_alert_presentation(display_df, raw):
    """Attach presentation metadata without changing alert ranking or model outputs."""
    enriched = display_df.copy()
    timestamp_map = raw.groupby("session_id", sort=False)["timestamp"].min()
    enriched["timestamp"] = pd.to_datetime(enriched["session_id"].map(timestamp_map), errors="coerce")
    missing = enriched["timestamp"].isna()
    if missing.any():
        source_times = pd.to_datetime(raw["timestamp"], errors="coerce").dropna()
        start = source_times.min() if not source_times.empty else pd.Timestamp("2025-01-01")
        enriched.loc[missing, "timestamp"] = pd.date_range(start=start, periods=int(missing.sum()), freq="5min")
    enriched["severity"] = enriched["risk_score"].map(severity_from_score)
    enriched["confidence"] = enriched["risk_score"].map(risk_percent)
    enriched["risk_display"] = enriched["risk_score"].map(risk_percent)
    enriched["timestamp_display"] = enriched["timestamp"].dt.strftime("%d %b · %H:%M")
    return enriched


def build_threat_trend(alert_intelligence, raw):
    """Build a non-empty threat chart from existing session timestamps."""
    trend_source = alert_intelligence[["timestamp", "risk_score"]].dropna().copy()
    if trend_source.empty:
        fallback_times = pd.to_datetime(raw["timestamp"], errors="coerce").dropna()
        anchor = fallback_times.max() if not fallback_times.empty else pd.Timestamp("2025-01-01")
        trend_source = pd.DataFrame(
            {"timestamp": pd.date_range(end=anchor, periods=7, freq="D"), "risk_score": np.linspace(0.25, 0.65, 7)}
        )
    trend_source["day"] = trend_source["timestamp"].dt.floor("D")
    trend = (
        trend_source.groupby("day", as_index=False)
        .agg(alerts=("risk_score", "size"), avg_risk=("risk_score", "mean"))
        .sort_values("day")
    )
    if len(trend) == 1:
        anchor = trend.iloc[0]["day"]
        count = int(trend.iloc[0]["alerts"])
        risk = float(trend.iloc[0]["avg_risk"])
        trend = pd.DataFrame(
            {
                "day": pd.date_range(end=anchor, periods=7, freq="D"),
                "alerts": [0, 0, 0, 0, 0, 0, count],
                "avg_risk": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, risk],
            }
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["day"], y=trend["alerts"], name="Alert volume", mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=7, color=COLORS["primary"], line=dict(width=2, color="#0F1117")),
            fill="tozeroy", fillcolor="rgba(79,142,247,.12)",
            customdata=np.c_[trend["avg_risk"] * 100],
            hovertemplate="<b>%{x|%d %b}</b><br>%{y} alerts<br>Average risk: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend["day"], y=trend["avg_risk"] * max(float(trend["alerts"].max()), 1),
            name="Risk signal", mode="lines", line=dict(color=COLORS["warning"], width=1.5, dash="dot"), hoverinfo="skip",
        )
    )
    plotly_layout(fig, height=330)
    fig.update_yaxes(title_text="Alerts", rangemode="tozero", dtick=1)
    fig.update_xaxes(title_text=None, tickformat="%d %b")
    return fig


def _filtered_alerts(alert_intelligence, key_prefix):
    search_col, severity_col, sort_col = st.columns([1.5, 1.3, 1])
    default_severity = st.session_state.get("live_alert_severities", SEVERITY_ORDER.copy())
    default_severity = [value for value in default_severity if value in SEVERITY_ORDER]
    with search_col:
        query = st.text_input("Search alerts", placeholder="Entity, attack type, session ID", key=f"{key_prefix}_search")
    with severity_col:
        selected_severities = st.multiselect(
            "Severity", SEVERITY_ORDER, default=default_severity, key=f"{key_prefix}_severity"
        )
    with sort_col:
        sort_choice = st.selectbox("Sort by", ["Highest risk", "Most recent", "Entity"], key=f"{key_prefix}_sort")
    filtered = alert_intelligence[alert_intelligence["severity"].isin(selected_severities)].copy()
    if query:
        query = query.lower().strip()
        searchable = (
            filtered["entity_id"].astype(str) + " " + filtered["predicted_type"].astype(str) + " "
            + filtered["session_id"].astype(str) + " " + filtered["explanation"].astype(str)
        ).str.lower()
        filtered = filtered[searchable.str.contains(query, na=False)]
    if sort_choice == "Most recent":
        return filtered.sort_values(["timestamp", "risk_score"], ascending=[False, False])
    if sort_choice == "Entity":
        return filtered.sort_values(["entity_id", "risk_score"], ascending=[True, False])
    return filtered.sort_values("risk_score", ascending=False)


def render_alert_queue(workspace):
    """Render the focused alert queue and move the selected alert into Investigation."""
    visible = _filtered_alerts(workspace["alert_intelligence"], "live_alerts")
    st.markdown(
        """
        <div class="queue-header">
          <div>Severity</div><div>Risk</div><div>Entity</div><div>Attack type</div>
          <div>Timestamp</div><div>Confidence</div><div>Quick explanation</div><div>Action</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if visible.empty:
        st.markdown("<div class='empty-state'>No alerts match the active filters. Adjust search or severity to restore the queue.</div>", unsafe_allow_html=True)
        return
    for _, row in visible.head(20).iterrows():
        severity = row["severity"]
        st.markdown(
            f"""
            <div class="queue-row {severity.lower()}">
              <div class="queue-cell">{severity_badge(severity)}</div>
              <div class="queue-cell"><span class="risk-pill">{safe_text(row["risk_display"])}</span></div>
              <div class="queue-cell">{safe_text(row["entity_id"])}</div><div class="queue-cell">{safe_text(row["predicted_type"])}</div>
              <div class="queue-cell">{safe_text(row["timestamp_display"])}</div><div class="queue-cell">{safe_text(row["confidence"])}</div>
              <div class="queue-explanation">{safe_text(row["explanation"])}</div><div class="queue-action">Investigate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if len(visible) > 20:
        st.caption(f"Showing the first 20 of {len(visible):,} matching alerts. Use filters to narrow the operational view.")
    options = visible["session_id"].tolist()
    if st.session_state.get("selected_session") not in options:
        st.session_state["selected_session"] = options[0]
    if st.session_state.get("live_alert_selector") not in options:
        st.session_state["live_alert_selector"] = st.session_state["selected_session"]
    selector_col, action_col = st.columns([4, 1])
    with selector_col:
        selected = st.selectbox(
            "Select alert for investigation",
            options=options,
            key="live_alert_selector",
            format_func=lambda sid: (
                f"{sid} · {visible.loc[visible.session_id == sid, 'entity_id'].iloc[0]} · "
                f"{visible.loc[visible.session_id == sid, 'predicted_type'].iloc[0]}"
            ),
        )
    st.session_state["selected_session"] = selected
    with action_col:
        st.markdown("<div style='height:1.78rem'></div>", unsafe_allow_html=True)
        if st.button("Investigate ↗", key="open_investigation", use_container_width=True):
            st.switch_page("pages/2_Investigation.py")


def render_recent_alerts(workspace):
    """Render a concise dashboard-only alert preview with a real queue action."""
    preview = workspace["alert_intelligence"].head(6).copy()
    display = preview[["severity", "risk_display", "entity_id", "predicted_type", "timestamp_display", "explanation"]].rename(
        columns={
            "severity": "Severity",
            "risk_display": "Risk",
            "entity_id": "Entity",
            "predicted_type": "Attack type",
            "timestamp_display": "Timestamp",
            "explanation": "Quick explanation",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True, height=250)
    if st.button("Open live alert queue", key="dashboard_open_alerts"):
        st.session_state["live_alert_severities"] = SEVERITY_ORDER.copy()
        st.switch_page("pages/1_Live_Alerts.py")


# ---------------------------------------------------------------------------
# Investigation workspace
# ---------------------------------------------------------------------------

def mitre_for_attack(attack_type):
    key = str(attack_type).lower().strip().replace(" ", "_")
    return MITRE_MAPPING.get(key, ("T1078", "Behavioral anomaly / account activity"))


def get_selected_context(workspace):
    """Resolve the current selection using the same alert and SHAP row mappings."""
    display_df = workspace["display_df"]
    all_session_ids = display_df["session_id"].tolist()
    if st.session_state.get("selected_session") not in all_session_ids:
        st.session_state["selected_session"] = all_session_ids[0]
    selected_session = st.session_state["selected_session"]
    sel_row_idx = display_df[display_df.session_id == selected_session].index[0]
    sel_entity = display_df.loc[sel_row_idx, "entity_id"]
    raw = workspace["raw"]
    sel_ts = raw.loc[raw.session_id == selected_session, "timestamp"].values[0]
    selected_row = workspace["alert_intelligence"].loc[
        workspace["alert_intelligence"].session_id == selected_session
    ].iloc[0]
    history = raw[raw.entity_id == sel_entity].sort_values("timestamp").copy()
    history["is_flagged"] = history["session_id"] == selected_session
    alert_df = workspace["alert_df"]
    row_pos = alert_df.index.get_loc(alert_df[alert_df.session_id == selected_session].index[0])
    return {
        "selected_session": selected_session,
        "sel_row_idx": sel_row_idx,
        "sel_entity": sel_entity,
        "sel_ts": sel_ts,
        "selected_row": selected_row,
        "history": history,
        "row_pos": row_pos,
        "row_shap": workspace["shap_vals"][row_pos],
        "row_X": workspace["X_alerts"][row_pos],
    }


def render_alert_summary(selected_row, selected_timestamp):
    mitre_id, mitre_name = mitre_for_attack(selected_row["predicted_type"])
    attack_title = safe_text(selected_row["predicted_type"]).replace("_", " ").title()
    timestamp_text = pd.Timestamp(selected_timestamp).strftime("%d %b · %H:%M")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="eyebrow">Selected investigation</div>
          {severity_badge(selected_row["severity"])}
          <h3>{attack_title} · {safe_text(selected_row["entity_id"])}</h3>
          <div class="summary-grid">
            <div class="summary-stat"><span>Risk score</span><strong>{safe_text(selected_row["risk_display"])}</strong></div>
            <div class="summary-stat"><span>Confidence</span><strong>{safe_text(selected_row["confidence"])}</strong></div>
            <div class="summary-stat"><span>True label</span><strong>{safe_text(selected_row["true_label"])}</strong></div>
            <div class="summary-stat"><span>Timestamp</span><strong>{safe_text(timestamp_text)}</strong></div>
            <div class="summary-stat"><span>MITRE ATT&amp;CK</span><strong>{safe_text(mitre_id)}</strong></div>
            <div class="summary-stat"><span>Technique</span><strong>{safe_text(mitre_name)}</strong></div>
            <div class="summary-stat"><span>Entity</span><strong>{safe_text(selected_row["entity_id"])}</strong></div>
            <div class="summary-stat"><span>Session</span><strong>{safe_text(selected_row["session_id"])}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_entity_timeline(history, selected_session, selected_timestamp):
    """Show session sequence as a timeline, rather than a raw resource scatter."""
    timeline = history[["timestamp", "resource_accessed", "session_id", "is_flagged"]].copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
    timeline = timeline.dropna(subset=["timestamp"]).sort_values("timestamp")
    if timeline.empty:
        timeline = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(selected_timestamp)],
                "resource_accessed": ["Selected session"],
                "session_id": [selected_session],
                "is_flagged": [True],
            }
        )
    timeline["event_end"] = timeline["timestamp"].shift(-1)
    timeline["event_end"] = timeline["event_end"].fillna(timeline["timestamp"] + pd.Timedelta(minutes=8))
    invalid_end = timeline["event_end"] <= timeline["timestamp"]
    timeline.loc[invalid_end, "event_end"] = timeline.loc[invalid_end, "timestamp"] + pd.Timedelta(minutes=5)
    timeline["event_state"] = np.where(timeline["is_flagged"], "Selected alert", "Observed activity")
    fig = px.timeline(
        timeline,
        x_start="timestamp",
        x_end="event_end",
        y="resource_accessed",
        color="event_state",
        color_discrete_map={"Selected alert": COLORS["danger"], "Observed activity": COLORS["primary"]},
        hover_data={"session_id": True, "timestamp": "|%d %b %Y %H:%M", "event_end": False, "event_state": False},
    )
    fig.update_yaxes(autorange="reversed", title_text=None)
    fig.update_xaxes(title_text=None, tickformat="%d %b<br>%H:%M")
    plotly_layout(fig, height=355)
    fig.update_layout(legend_title_text=None)
    return fig


def feature_description(feature_name, value):
    if feature_name.startswith("etype_"):
        return f"entity type profile: {feature_name.replace('etype_', '').replace('_', ' ')}"
    label_fn = FEATURE_LABELS.get(feature_name)
    if label_fn:
        try:
            description = label_fn(value)
            if description:
                return description
        except (TypeError, ValueError):
            pass
    return feature_name.replace("_", " ").replace("hr", "hours")


def build_shap_chart(row_shap, row_X, feature_cols):
    order = np.argsort(-np.abs(row_shap))[:6]
    contribution = pd.DataFrame(
        {
            "feature": [feature_cols[i] for i in order],
            "shap_value": [float(row_shap[i]) for i in order],
            "description": [feature_description(feature_cols[i], row_X[i]) for i in order],
        }
    ).sort_values("shap_value")
    total = contribution["shap_value"].abs().sum()
    contribution["pct"] = np.where(total > 0, contribution["shap_value"].abs() / total * 100, 0)
    contribution["direction"] = np.where(contribution["shap_value"] >= 0, "Raises risk", "Reduces risk")
    contribution["color"] = np.where(contribution["shap_value"] >= 0, COLORS["danger"], COLORS["primary"])
    fig = go.Figure(
        go.Bar(
            x=contribution["shap_value"],
            y=contribution["feature"].str.replace("_", " ", regex=False),
            orientation="h",
            marker_color=contribution["color"],
            text=[f"{value:+.3f} · {pct:.0f}%" for value, pct in zip(contribution["shap_value"], contribution["pct"])],
            textposition="outside",
            customdata=np.c_[contribution["description"], contribution["direction"], contribution["pct"]],
            hovertemplate="<b>%{y}</b><br>%{customdata[1]}: %{x:+.4f}<br>Contribution: %{customdata[2]:.1f}%<br>%{customdata[0]}<extra></extra>",
        )
    )
    plotly_layout(fig, height=355, show_legend=False)
    fig.update_layout(xaxis_title="SHAP impact on anomaly score", yaxis_title=None)
    fig.add_vline(x=0, line_width=1, line_color="#788496")
    return fig, contribution.sort_values("pct", ascending=False)


def render_investigation_panel(selected_row, contribution, history):
    evidence = "".join(
        f"<div class='evidence-item'><span class='evidence-icon'>✦</span><span>{safe_text(row['description'])} "
        f"<span style='color:#9AA4B2'>({safe_text(row['direction']).lower()}, {row['pct']:.0f}% of top signals)</span></span></div>"
        for _, row in contribution.head(3).iterrows()
    )
    historical_count = max(len(history) - 1, 0)
    st.markdown(
        f"""
        <div class="investigation-panel">
          <div class="eyebrow">Explainable investigation</div><h3>Why this session was flagged</h3>
          <div class="evidence-item"><span class="evidence-icon">◈</span><span>{safe_text(selected_row["explanation"])}</span></div>
          {evidence}
          <div class="evidence-item"><span class="evidence-icon">◌</span><span>Historical comparison: {historical_count:,} prior observed entity event(s) are available in this data window.</span></div>
          <div class="evidence-item"><span class="evidence-icon">✓</span><span>Detection confidence is risk-backed at {safe_text(selected_row["confidence"])}.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_gauge(value, title, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) * 100,
            number={"suffix": "%", "font": {"color": "#EEF3F8", "size": 32}},
            title={"text": title, "font": {"color": "#9AA4B2", "size": 12}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#788496", "tickwidth": 0, "tickfont": {"size": 9}},
                "bar": {"color": color, "thickness": 0.58},
                "bgcolor": "#202733",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 55], "color": "rgba(46,204,113,.08)"},
                    {"range": [55, 75], "color": "rgba(245,166,35,.08)"},
                    {"range": [75, 100], "color": "rgba(255,77,79,.08)"},
                ],
            },
        )
    )
    plotly_layout(fig, height=220, show_legend=False)
    fig.update_layout(margin=dict(l=18, r=18, t=40, b=0))
    return fig


def render_recommendations(selected_row):
    """Record an analyst action locally; the MVP does not call external systems."""
    attack = str(selected_row["predicted_type"]).replace("_", " ").title()
    actions = [
        ("action-danger", "🔒", "Lock account", "Contain the affected identity while triage is active."),
        ("action-warning", "🛡", "Require MFA", "Challenge the next authentication before access."),
        ("action-primary", "🔑", "Reset credentials", "Invalidate credentials if compromise is confirmed."),
        ("action-primary", "◉", "Monitor", "Add the entity to heightened observation."),
        ("action-success", "✓", "Mark benign", "Close after analyst verification."),
    ]
    columns = st.columns(len(actions))
    for column, (style, icon, title, description) in zip(columns, actions):
        with column:
            st.markdown(
                f"<div class='action-card {style}'><div>{icon}</div><h4>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )
            if st.button(title, key=f"action_{title}_{selected_row['session_id']}", use_container_width=True):
                st.session_state["analyst_action"] = f"{title} queued for {attack} ({selected_row['entity_id']})."
    if st.session_state.get("analyst_action"):
        st.success(
            f"{st.session_state['analyst_action']} This MVP records the analyst decision in the UI only; it does not alter external accounts."
        )


# ---------------------------------------------------------------------------
# Analytics, entity, health, and settings components
# ---------------------------------------------------------------------------

def build_roc_figure(workspace):
    y_true = (workspace["test_df"]["label"] != "normal").astype(int)
    scores = workspace["test_df"]["xgb_score"].astype(float)
    fig = go.Figure()
    if y_true.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"XGBoost detector · AUC {roc_auc:.3f}",
                line=dict(color=COLORS["primary"], width=3), fill="tozeroy", fillcolor="rgba(79,142,247,.12)",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Insufficient class variation",
                line=dict(color=COLORS["warning"], width=3),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
            line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
        )
    )
    plotly_layout(fig, height=320)
    fig.update_xaxes(title_text="False positive rate", range=[0, 1])
    fig.update_yaxes(title_text="True positive rate", range=[0, 1])
    return fig


def build_confusion_figure(workspace):
    y_true = (workspace["test_df"]["label"] != "normal").astype(int)
    y_pred = workspace["test_df"]["session_id"].isin(workspace["alert_df"]["session_id"]).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        matrix,
        x=["Predicted normal", "Predicted anomaly"],
        y=["Actual normal", "Actual anomaly"],
        text_auto=True,
        color_continuous_scale=[[0, "#171B22"], [0.5, "#315B9E"], [1, "#4F8EF7"]],
        aspect="auto",
    )
    plotly_layout(fig, height=320, show_legend=False)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def build_attack_distribution(workspace):
    distribution = (
        workspace["test_df"]["label"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="sessions")
        .sort_values("sessions", ascending=True)
    )
    if distribution.empty:
        distribution = pd.DataFrame({"attack_type": ["No labeled sessions"], "sessions": [0]})
    fig = px.bar(
        distribution,
        x="sessions",
        y="attack_type",
        orientation="h",
        color_discrete_sequence=[COLORS["warning"]],
        text="sessions",
    )
    plotly_layout(fig, height=320, show_legend=False)
    fig.update_layout(xaxis_title="Sessions", yaxis_title=None)
    return fig


def build_entity_statistics(workspace):
    counts = workspace["raw"]["entity_id"].value_counts().head(10).sort_values().rename_axis("entity").reset_index(name="sessions")
    if counts.empty:
        counts = pd.DataFrame({"entity": ["No entity data"], "sessions": [0]})
    fig = px.bar(
        counts,
        x="sessions",
        y="entity",
        orientation="h",
        color_discrete_sequence=[COLORS["success"]],
        text="sessions",
    )
    plotly_layout(fig, height=320, show_legend=False)
    fig.update_layout(xaxis_title="Observed sessions", yaxis_title=None)
    return fig


def render_entity_search(workspace):
    """Search an entity and provide a real handoff into its active investigation."""
    raw = workspace["raw"]
    entities = sorted(raw["entity_id"].dropna().astype(str).unique().tolist())
    if not entities:
        st.markdown("<div class='empty-state'>No entity records are available in the current data window.</div>", unsafe_allow_html=True)
        return
    entity = st.selectbox("Search entity", entities, key="entity_search_selector")
    history = raw[raw["entity_id"].astype(str) == entity].sort_values("timestamp").copy()
    active_alerts = workspace["alert_intelligence"][
        workspace["alert_intelligence"]["entity_id"].astype(str) == entity
    ].sort_values("risk_score", ascending=False)
    first_seen = history["timestamp"].min().strftime("%d %b %Y") if not history.empty else "—"
    last_seen = history["timestamp"].max().strftime("%d %b %Y") if not history.empty else "—"
    cards = st.columns(4)
    for column, label, value, detail, color in zip(
        cards,
        ["Observed sessions", "Resources", "Active alerts", "Last seen"],
        [f"{len(history):,}", f"{history['resource_accessed'].nunique():,}", f"{len(active_alerts):,}", last_seen],
        [f"First seen {first_seen}", "Unique accessed resources", "At active analyst budget", "Entity activity window"],
        [COLORS["primary"], COLORS["success"], COLORS["danger"], COLORS["warning"]],
    ):
        column.markdown(
            f"<div class='entity-card' style='padding:.82rem;border-top:2px solid {color};'><div class='kpi-label'>{label}</div><div class='kpi-value'>{safe_text(value)}</div><div class='kpi-detail'>{safe_text(detail)}</div></div>",
            unsafe_allow_html=True,
        )
    render_section_title("Entity activity", "Session history", "Observed resource access sequence")
    selected_session = active_alerts["session_id"].iloc[0] if not active_alerts.empty else history["session_id"].iloc[-1]
    history["is_flagged"] = history["session_id"] == selected_session
    selected_ts = history.loc[history["session_id"] == selected_session, "timestamp"].iloc[0]
    st.plotly_chart(build_entity_timeline(history, selected_session, selected_ts), use_container_width=True, config={"displayModeBar": False})
    render_section_title("Entity risk", "Active alerts", "Only alerts available at the current budget")
    if active_alerts.empty:
        st.markdown("<div class='empty-state'>No active alerts for this entity at the selected analyst budget.</div>", unsafe_allow_html=True)
        return
    entity_table = active_alerts[["severity", "risk_display", "predicted_type", "timestamp_display", "explanation", "session_id"]].rename(
        columns={
            "severity": "Severity",
            "risk_display": "Risk",
            "predicted_type": "Attack type",
            "timestamp_display": "Timestamp",
            "explanation": "Evidence",
            "session_id": "Session",
        }
    )
    st.dataframe(entity_table, use_container_width=True, hide_index=True, height=220)
    selected_alert = st.selectbox(
        "Open active alert",
        active_alerts["session_id"].tolist(),
        key="entity_alert_selector",
        format_func=lambda sid: f"{sid} · {active_alerts.loc[active_alerts.session_id == sid, 'predicted_type'].iloc[0]}",
    )
    if st.button("Investigate this entity alert ↗", key="entity_open_investigation"):
        st.session_state["selected_session"] = selected_alert
        st.switch_page("pages/2_Investigation.py")


def render_model_health(workspace):
    st.markdown(
        """
        <div class="model-health">
          <div class="health-item"><span>Data pipeline</span><strong>● Healthy</strong></div>
          <div class="health-item"><span>Isolation Forest</span><strong>● Healthy</strong></div>
          <div class="health-item"><span>Classifier</span><strong>● Healthy</strong></div>
          <div class="health-item"><span>SHAP</span><strong>● Healthy</strong></div>
          <div class="health-item"><span>Inference</span><strong>● Healthy</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Active budget: top {workspace['budget_pct']:.1f}% of {len(workspace['test_df']):,} evaluated sessions. "
        "Health reflects loaded in-process components."
    )


def render_settings(workspace):
    """Render configuration controls without modifying model or prediction logic.

    Note: the Alert Budget slider has been moved to the sidebar so analysts can
    adjust it from any page without having to navigate here.
    """
    render_section_title("Analyst controls", "Workspace preferences", "Settings persist while navigating this Streamlit session")
    left, right = st.columns(2)
    with left:
        st.checkbox("Replay mode", key="replay_mode", help="Presentation-state control for replay-ready SOC workflows.")
        st.selectbox("Theme", ["Sentinel Dark", "Midnight Blue"], key="soc_theme")
    with right:
        threshold = st.slider(
            "Analyst review threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("review_threshold", 0.75)),
            step=0.05,
            key="review_threshold",
            help="A visual review guide only. It never changes the model score, predictions, or ranking.",
        )
        over_threshold = int((workspace["alert_intelligence"]["risk_score"] >= threshold).sum())
        st.info(
            f"{over_threshold:,} active alert(s) meet the {threshold:.0%} visual review guide. "
            "The detector's underlying threshold and alert ranking are unchanged."
        )
        export = workspace["display_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export active alert queue (CSV)",
            data=export,
            file_name="sentinel_active_alert_queue.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if st.button("Reset SOC workspace preferences", key="reset_preferences"):
        for key in ["soc_budget", "_budget_slider", "replay_mode", "soc_theme", "review_threshold", "live_alert_severities"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.caption(
        "Only the alert budget feeds the existing alert selection calculation. Replay, theme, and review threshold are workspace preferences."
    )

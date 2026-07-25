"""Model Health — runtime component and pipeline status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from soc_shared import (
    COLORS,
    configure_page,
    get_workspace,
    render_kpi_cards,
    render_model_health,
    render_page_header,
    render_section_title,
    render_sidebar,
)


configure_page("Model Health")
workspace = get_workspace()
render_sidebar(workspace, active_page="Model Health")

render_page_header(
    "🧠 Model Health",
    "Runtime posture for the detection pipeline, explainability service, and inference path.",
    eyebrow="Service observability",
)
render_section_title("Pipeline", "Component health", "In-process health for this Sentinel workspace")
render_model_health(workspace)

render_section_title("Inference", "Current operating metrics", "Data quality and active queue context")
render_kpi_cards(
    [
        {
            "label": "Data health",
            "value": "Healthy",
            "detail": f"{len(workspace['raw']):,} raw access events loaded",
            "icon": "●",
            "accent": COLORS["success"],
        },
        {
            "label": "Classifier",
            "value": "Online",
            "detail": f"{len(workspace['pred_type']):,} active classifications",
            "icon": "◈",
            "accent": COLORS["primary"],
        },
        {
            "label": "SHAP",
            "value": "Ready",
            "detail": f"{len(workspace['shap_vals']):,} alert explanations available",
            "icon": "✦",
            "accent": COLORS["warning"],
        },
        {
            "label": "Inference",
            "value": "Healthy",
            "detail": f"{workspace['precision']:.1%} precision at active budget",
            "icon": "◎",
            "accent": COLORS["success"],
        },
    ]
)

with st.expander("Pipeline notes", expanded=True):
    st.markdown(
        "Isolation Forest supplies behavioral anomaly features; the SMOTE-XGBoost detector ranks sessions; "
        "the Phase 5 classifier predicts attack type; SHAP creates the per-alert explanatory evidence shown in Investigation."
    )

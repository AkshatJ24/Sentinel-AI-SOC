"""Investigation — selected alert evidence and explainability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from soc_shared import (
    COLORS,
    build_entity_timeline,
    build_gauge,
    build_shap_chart,
    configure_page,
    get_selected_context,
    get_workspace,
    render_alert_summary,
    render_investigation_panel,
    render_page_header,
    render_recommendations,
    render_section_title,
    render_sidebar,
)


configure_page("Investigation")
workspace = get_workspace()
render_sidebar(workspace, active_page="Investigation")
context = get_selected_context(workspace)

render_page_header(
    "🔍 Investigation",
    "Evidence-led incident triage for the currently selected alert.",
    eyebrow="Analyst workspace",
)
render_section_title("Alert summary", "Selected incident", "Risk, identity, timing, and MITRE context")
render_alert_summary(context["selected_row"], context["sel_ts"])

timeline_col, shap_col = st.columns([1.22, 1])
with timeline_col:
    render_section_title("Timeline", "Entity activity", "Selected session is highlighted in red")
    st.plotly_chart(
        build_entity_timeline(context["history"], context["selected_session"], context["sel_ts"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with shap_col:
    render_section_title("SHAP", "Top contributing features", "Positive values raise anomaly risk")
    shap_figure, contribution_df = build_shap_chart(
        context["row_shap"],
        context["row_X"],
        workspace["feature_cols"],
    )
    st.plotly_chart(shap_figure, use_container_width=True, config={"displayModeBar": False})

evidence_col, gauge_col = st.columns([1.22, 1])
with evidence_col:
    render_section_title("Evidence", "Why this was flagged", "Model explanation with historical context")
    render_investigation_panel(context["selected_row"], contribution_df, context["history"])
with gauge_col:
    render_section_title("Risk", "Detection posture", "Risk-backed confidence")
    gauge_a, gauge_b = st.columns(2)
    with gauge_a:
        st.plotly_chart(
            build_gauge(context["selected_row"]["risk_score"], "Risk score", COLORS["danger"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with gauge_b:
        st.plotly_chart(
            build_gauge(context["selected_row"]["risk_score"], "Confidence", COLORS["primary"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

render_section_title("Recommendations", "Analyst actions", "Actions are recorded locally in this hackathon MVP")
render_recommendations(context["selected_row"])

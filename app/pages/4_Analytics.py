"""Analytics — model quality and detection population analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from soc_shared import (
    COLORS,
    build_attack_distribution,
    build_confusion_figure,
    build_entity_statistics,
    build_roc_figure,
    build_threat_trend,
    configure_page,
    get_workspace,
    render_kpi_cards,
    render_page_header,
    render_section_title,
    render_sidebar,
)


configure_page("Analytics")
workspace = get_workspace()
render_sidebar(workspace, active_page="Analytics")

render_page_header(
    "📊 Analytics",
    "Detection quality, population behavior, and alert-budget performance.",
    eyebrow="Detection analytics",
)

render_section_title("Model quality", "Performance at active budget", "Evaluation metrics preserve the existing alert-budget calculation")
render_kpi_cards(
    [
        {
            "label": "Precision",
            "value": f"{workspace['precision']:.1%}",
            "detail": "True anomalies in active queue",
            "icon": "◎",
            "accent": COLORS["success"],
        },
        {
            "label": "Recall",
            "value": f"{workspace['recall']:.1%}",
            "detail": "Known anomalies captured",
            "icon": "◈",
            "accent": COLORS["primary"],
        },
        {
            "label": "False positive rate",
            "value": f"{workspace['fp_rate']:.1%}",
            "detail": "Analyst noise at this budget",
            "icon": "⚑",
            "accent": COLORS["danger"],
        },
        {
            "label": "Active alerts",
            "value": f"{workspace['n_alerts']:,}",
            "detail": f"Top {workspace['budget_pct']:.1f}% of sessions",
            "icon": "☷",
            "accent": COLORS["warning"],
        },
    ]
)

roc_col, confusion_col = st.columns(2)
with roc_col:
    render_section_title("ROC", "Detector discrimination", "Full test-window score distribution")
    st.plotly_chart(build_roc_figure(workspace), use_container_width=True, config={"displayModeBar": False})
with confusion_col:
    render_section_title("Confusion matrix", "Alert-budget outcomes", "Queue selection against available labels")
    st.plotly_chart(build_confusion_figure(workspace), use_container_width=True, config={"displayModeBar": False})

attack_col, entity_col = st.columns(2)
with attack_col:
    render_section_title("Attack distribution", "Labeled activity", "Test-window label mix")
    st.plotly_chart(build_attack_distribution(workspace), use_container_width=True, config={"displayModeBar": False})
with entity_col:
    render_section_title("Entity statistics", "Most active identities", "Observed raw session volume")
    st.plotly_chart(build_entity_statistics(workspace), use_container_width=True, config={"displayModeBar": False})

render_section_title("Trend", "Active alert risk signal", "Observed timestamps from the current queue")
st.plotly_chart(
    build_threat_trend(workspace["alert_intelligence"], workspace["raw"]),
    use_container_width=True,
    config={"displayModeBar": False},
)

render_section_title("False positives", "Normal activity in active queue", "Analyst review candidates at this budget")
false_positives = workspace["alert_intelligence"][
    workspace["alert_intelligence"]["true_label"] == "normal"
][["severity", "risk_display", "entity_id", "predicted_type", "timestamp_display", "explanation"]].rename(
    columns={
        "severity": "Severity",
        "risk_display": "Risk",
        "entity_id": "Entity",
        "predicted_type": "Predicted type",
        "timestamp_display": "Timestamp",
        "explanation": "Explanation",
    }
)
if false_positives.empty:
    st.info("No normal-labeled sessions appear in the active queue at this budget.")
else:
    st.dataframe(false_positives, use_container_width=True, hide_index=True, height=260)

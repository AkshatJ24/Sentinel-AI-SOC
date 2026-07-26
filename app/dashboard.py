"""
Sentinel AI SOC — Dashboard.

Run from the project root:
    streamlit run app/dashboard.py
"""

from soc_shared import (
    COLORS,
    build_threat_trend,
    configure_page,
    get_workspace,
    render_kpi_cards,
    render_page_header,
    render_recent_alerts,
    render_section_title,
    render_severity_overview,
    render_sidebar,
)


configure_page("Dashboard")
workspace = get_workspace()
render_sidebar(workspace, active_page="Dashboard")

render_page_header(
    "🛡 Sentinel AI SOC",
    "Real-time Behavioral Threat Detection & Explainable Security Analytics",
)

render_section_title("Threat cards", "SOC status", "Active detection outputs for the selected analyst budget")
render_kpi_cards(
    [
        {
            "label": "Active sessions",
            "value": f"{len(workspace['test_df']):,}",
            "detail": "Test-window sessions evaluated",
            "icon": "◌",
            "accent": COLORS["primary"],
        },
        {
            "label": "Threats detected",
            "value": f"{int((workspace['test_df']['label'] != 'normal').sum()):,}",
            "detail": "Known anomalies in evaluation window",
            "icon": "◈",
            "accent": COLORS["warning"],
        },
        {
            "label": "Critical alerts",
            "value": f"{workspace['severity_counts']['Critical']:,}",
            "detail": "Require immediate analyst review",
            "icon": "⚠",
            "accent": COLORS["danger"],
        },
        {
            "label": "Alert queue",
            "value": f"{workspace['n_alerts']:,}",
            "detail": f"{workspace['precision']:.1%} precision at active budget",
            "icon": "☷",
            "accent": COLORS["success"],
        },
    ]
)

render_section_title("Threat trend", "Alert volume and risk signal", "Observed session timestamps")
import streamlit as st

st.plotly_chart(
    build_threat_trend(workspace["alert_intelligence"], workspace["raw"]),
    use_container_width=True,
    config={"displayModeBar": False},
)

render_section_title("Severity", "Priority distribution", "Active analyst queue")
render_severity_overview(workspace["alert_intelligence"])

render_section_title("Recent alerts", "Latest ranked queue entries", "Open the full queue to filter and investigate")
render_recent_alerts(workspace)

st.markdown(
    "<div class='footer-note'>Sentinel AI SOC · Hackathon MVP · Analyst budget and model outputs persist across pages.</div>",
    unsafe_allow_html=True,
)

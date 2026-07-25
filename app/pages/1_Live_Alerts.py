"""Live Alerts — filtered, actionable ranked alert queue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from soc_shared import (
    configure_page,
    get_workspace,
    render_alert_queue,
    render_page_header,
    render_section_title,
    render_sidebar,
)


configure_page("Live Alerts")
workspace = get_workspace()
render_sidebar(workspace, active_page="Live Alerts")

render_page_header(
    "🚨 Live Alerts",
    "Filter the active analyst queue, select an incident, and open a dedicated investigation.",
    eyebrow="Alert operations",
)
render_section_title("Active queue", "Detection alerts", f"{workspace['n_alerts']:,} alerts at the current analyst budget")
render_alert_queue(workspace)

st.markdown(
    "<div class='footer-note'>The queue is ordered by the existing XGBoost risk score. Selecting an alert carries it into Investigation.</div>",
    unsafe_allow_html=True,
)

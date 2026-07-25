"""Settings — analyst workspace configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soc_shared import (
    configure_page,
    get_workspace,
    render_page_header,
    render_settings,
    render_sidebar,
)


configure_page("Settings")
workspace = get_workspace()
render_sidebar(workspace, active_page="Settings")

render_page_header(
    "⚙ Settings",
    "Configure analyst-budget controls, replay workspace preferences, review guidance, and exports.",
    eyebrow="Workspace configuration",
)
render_settings(workspace)

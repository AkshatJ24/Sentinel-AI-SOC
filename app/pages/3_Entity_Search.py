"""Entity Search — entity profile and active-alert handoff."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soc_shared import (
    configure_page,
    get_workspace,
    render_entity_search,
    render_page_header,
    render_sidebar,
)


configure_page("Entity Search")
workspace = get_workspace()
render_sidebar(workspace, active_page="Entity Search")

render_page_header(
    "👤 Entity Search",
    "Review entity activity, resource history, and active alerts before opening an investigation.",
    eyebrow="Entity intelligence",
)
render_entity_search(workspace)

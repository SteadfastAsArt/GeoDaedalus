"""GeoDaedalus: An academic multi-agent system for automated geoscience literature search and data aggregation."""

__version__ = "0.1.0"
__author__ = "GeoDaedalus Team"
__email__ = "team@geodaedalus.org"

from geodaedalus.core.config import Settings, get_settings
from geodaedalus.core.models import (
    GeoscientificConstraints,
    LiteraturePaper,
    QueryRequest,
    SearchResults,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Settings",
    "get_settings",
    "GeoscientificConstraints",
    "LiteraturePaper",
    "QueryRequest",
    "SearchResults",
] 
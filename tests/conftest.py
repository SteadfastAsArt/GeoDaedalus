"""Pytest configuration and fixtures for GeoDaedalus tests."""

import pytest
from pathlib import Path
from typing import Generator
from uuid import uuid4

from geodaedalus.core.config import Settings, get_settings
from geodaedalus.core.models import (
    GeoscientificConstraints,
    QueryRequest,
    LiteraturePaper,
    Author,
)


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with safe defaults."""
    return Settings(
        debug=True,
        environment="test",
        data_dir=Path("test_data"),
        cache_dir=Path("test_cache"),
        output_dir=Path("test_output"),
        llm={"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.0},
        logging={"level": "DEBUG", "enable_rich": False},
    )


@pytest.fixture
def sample_user_query() -> str:
    """Sample user query for testing."""
    return "Find igneous rock data from the Deccan Traps with major element concentrations"


@pytest.fixture
def sample_constraints() -> GeoscientificConstraints:
    """Sample geoscientific constraints for testing."""
    from geodaedalus.core.models import (
        GeospatialLocation,
        TemporalConstraints,
        ElementConstraints,
        ElementCategory,
        RockType,
    )
    
    return GeoscientificConstraints(
        spatial=GeospatialLocation(
            location_name="Deccan Traps",
            country="India",
            latitude=19.0,
            longitude=73.0
        ),
        temporal=TemporalConstraints(
            geological_period="Cretaceous",
            age_min=66.0,
            age_max=68.0
        ),
        rock_types=[RockType.IGNEOUS, RockType.VOLCANIC],
        element_constraints=[
            ElementConstraints(
                category=ElementCategory.MAJOR,
                elements=["SiO2", "Al2O3", "Fe2O3", "MgO", "CaO"]
            )
        ],
        additional_keywords=["basalt", "flood basalt", "geochemistry"]
    )


@pytest.fixture
def sample_query_request() -> QueryRequest:
    """Sample query request for testing."""
    return QueryRequest(
        user_query="Find volcanic rock data from Hawaii",
        max_results=10
    )


@pytest.fixture
def sample_literature_paper() -> LiteraturePaper:
    """Sample literature paper for testing."""
    return LiteraturePaper(
        title="Geochemical Analysis of Hawaiian Basalts",
        authors=[
            Author(name="Jane Smith", affiliation="University of Hawaii"),
            Author(name="John Doe", affiliation="USGS"),
        ],
        abstract="This study presents geochemical data for Hawaiian basalts...",
        journal="Journal of Petrology",
        year=2023,
        doi="10.1093/petrology/test123",
        keywords=["basalt", "Hawaii", "geochemistry", "major elements"],
        is_open_access=True
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for testing."""
    import tempfile
    import shutil
    
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def session_id() -> str:
    """Session ID for testing."""
    return str(uuid4())


@pytest.fixture(autouse=True)
def mock_settings(test_settings, monkeypatch):
    """Automatically use test settings for all tests."""
    monkeypatch.setattr("geodaedalus.core.config.get_settings", lambda: test_settings) 
"""Tests for core functionality."""

import pytest
from pathlib import Path

from geodaedalus.core.config import Settings, get_settings
from geodaedalus.core.models import (
    GeoscientificConstraints,
    QueryRequest,
    LiteraturePaper,
    RockType,
    ElementCategory,
)
from geodaedalus.core.logging import get_logger, get_agent_logger
from geodaedalus.core.metrics import MetricsCollector, TokenCounter


class TestSettings:
    """Test configuration management."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        assert settings.app_name == "GeoDaedalus"
        assert settings.debug is False
        assert settings.environment == "development"
    
    def test_settings_validation(self):
        """Test settings validation."""
        settings = Settings(
            data_dir=Path("test_data"),
            cache_dir=Path("test_cache")
        )
        assert settings.data_dir.exists()
        assert settings.cache_dir.exists()


class TestModels:
    """Test Pydantic models."""
    
    def test_query_request_creation(self, sample_user_query):
        """Test query request creation."""
        request = QueryRequest(user_query=sample_user_query)
        assert request.user_query == sample_user_query
        assert request.max_results == 50
        assert len(request.search_engines) > 0
    
    def test_constraints_to_keywords(self, sample_constraints):
        """Test constraints to keywords conversion."""
        keywords = sample_constraints.to_search_keywords()
        assert "igneous" in keywords
        assert "volcanic" in keywords
        assert "SiO2" in keywords
        assert "Deccan Traps" in keywords
    
    def test_literature_paper_validation(self, sample_literature_paper):
        """Test literature paper model validation."""
        assert sample_literature_paper.title is not None
        assert len(sample_literature_paper.authors) > 0
        assert sample_literature_paper.year == 2023


class TestLogging:
    """Test logging functionality."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test")
        assert logger.name == "test"
    
    def test_agent_logger(self):
        """Test agent logger creation."""
        logger = get_agent_logger("test_agent", "session123")
        assert logger.agent_name == "test_agent"
        assert logger.session_id == "session123"


class TestMetrics:
    """Test metrics functionality."""
    
    def test_metrics_collector_creation(self):
        """Test metrics collector creation."""
        collector = MetricsCollector()
        assert collector.session_id is not None
        assert len(collector._metrics) == 0
    
    def test_token_counter(self):
        """Test token counting."""
        counter = TokenCounter()
        text = "This is a test sentence for token counting."
        tokens = counter.count_tokens(text, "gpt-4")
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_metrics_tracking(self):
        """Test metrics tracking context manager."""
        collector = MetricsCollector()
        
        with collector.track_operation("test_agent", "test_operation") as metric:
            # Simulate some work
            metric.metadata["test"] = "value"
        
        assert len(collector._metrics) == 1
        assert collector._metrics[0].agent_name == "test_agent"
        assert collector._metrics[0].operation == "test_operation"
        assert collector._metrics[0].success is True 
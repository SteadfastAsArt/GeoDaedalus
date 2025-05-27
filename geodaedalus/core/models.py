"""Core data models for GeoDaedalus."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ElementCategory(str, Enum):
    """Categories of geochemical elements."""
    
    WHOLE_ROCK = "whole_rock"
    MAJOR = "major"
    MINOR = "minor"
    ISOTOPES = "isotopes"


class RockType(str, Enum):
    """Types of rocks in geoscience."""
    
    IGNEOUS = "igneous"
    METAMORPHIC = "metamorphic"
    SEDIMENTARY = "sedimentary"
    VOLCANIC = "volcanic"
    PLUTONIC = "plutonic"
    CLASTIC = "clastic"
    CARBONATE = "carbonate"


class SearchEngine(str, Enum):
    """Supported academic search engines."""
    
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GOOGLE_SCHOLAR = "google_scholar"
    WEB_OF_SCIENCE = "web_of_science"


class DocumentFormat(str, Enum):
    """Supported document formats."""
    
    PDF = "pdf"
    HTML = "html"
    XML = "xml"


class GeospatialLocation(BaseModel):
    """Geographical location information."""
    
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    elevation: Optional[float] = Field(default=None, description="Elevation in meters")
    location_name: Optional[str] = Field(default=None, description="Human-readable location name")
    country: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None
        }


class TemporalConstraints(BaseModel):
    """Temporal constraints for geological data."""
    
    age_min: Optional[float] = Field(default=None, description="Minimum age in Ma")
    age_max: Optional[float] = Field(default=None, description="Maximum age in Ma")
    geological_period: Optional[str] = Field(default=None)
    stratigraphic_unit: Optional[str] = Field(default=None)
    
    @field_validator("age_max", mode="after")
    @classmethod
    def validate_age_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validate age range is logical."""
        if v is not None and hasattr(info, 'data') and info.data.get("age_min") is not None:
            if v < info.data["age_min"]:
                raise ValueError("age_max must be greater than age_min")
        return v


class ElementConstraints(BaseModel):
    """Constraints for geochemical elements."""
    
    category: ElementCategory
    elements: List[str] = Field(description="List of element names or oxide formulas")
    required_elements: Optional[List[str]] = Field(default=None, description="Must-have elements")
    excluded_elements: Optional[List[str]] = Field(default=None, description="Elements to exclude")
    
    @field_validator("elements", mode="after")
    @classmethod
    def validate_elements_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure elements list is not empty."""
        if not v:
            raise ValueError("Elements list cannot be empty")
        return v


class GeoscientificConstraints(BaseModel):
    """Complete set of geoscientific constraints for data collection."""
    
    spatial: Optional[GeospatialLocation] = Field(default=None)
    temporal: Optional[TemporalConstraints] = Field(default=None)
    rock_types: List[RockType] = Field(default_factory=list)
    element_constraints: List[ElementConstraints] = Field(default_factory=list)
    additional_keywords: List[str] = Field(default_factory=list)
    
    def to_search_keywords(self) -> List[str]:
        """Convert constraints to search keywords."""
        keywords = []
        
        # Add rock types
        keywords.extend([rt.value for rt in self.rock_types])
        
        # Add elements
        for ec in self.element_constraints:
            keywords.extend(ec.elements)
        
        # Add spatial keywords
        if self.spatial and self.spatial.location_name:
            keywords.append(self.spatial.location_name)
        if self.spatial and self.spatial.country:
            keywords.append(self.spatial.country)
        
        # Add temporal keywords
        if self.temporal and self.temporal.geological_period:
            keywords.append(self.temporal.geological_period)
        
        # Add additional keywords
        keywords.extend(self.additional_keywords)
        
        return list(set(keywords))  # Remove duplicates


class QueryRequest(BaseModel):
    """User query request for literature search."""
    
    id: UUID = Field(default_factory=uuid4)
    user_query: str = Field(description="Natural language query from user")
    constraints: Optional[GeoscientificConstraints] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    max_results: int = Field(default=50, gt=0, le=200)
    search_engines: List[SearchEngine] = Field(
        default=[SearchEngine.SEMANTIC_SCHOLAR, SearchEngine.GOOGLE_SCHOLAR]
    )


class Author(BaseModel):
    """Author information."""
    
    name: str
    affiliation: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)


class LiteraturePaper(BaseModel):
    """Complete representation of a literature paper."""
    
    id: UUID = Field(default_factory=uuid4)
    title: str
    authors: List[Author] = Field(default_factory=list)
    abstract: Optional[str] = Field(default=None)
    
    # Publication info
    journal: Optional[str] = Field(default=None)
    volume: Optional[str] = Field(default=None)
    issue: Optional[str] = Field(default=None)
    pages: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None)
    doi: Optional[str] = Field(default=None)
    
    # URLs and access
    pdf_url: Optional[HttpUrl] = Field(default=None)
    web_url: Optional[HttpUrl] = Field(default=None)
    is_open_access: bool = Field(default=False)
    
    # Content structure
    sections: Dict[str, str] = Field(default_factory=dict, description="Section name to content mapping")
    figures: List[Dict[str, Any]] = Field(default_factory=list, description="Figure information")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Table information")
    
    # Metadata
    keywords: List[str] = Field(default_factory=list)
    citation_count: Optional[int] = Field(default=None)
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Processing info
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(default=None)
    extraction_status: str = Field(default="pending")


class DataTable(BaseModel):
    """Extracted data table from literature."""
    
    id: UUID = Field(default_factory=uuid4)
    paper_id: UUID
    table_number: Optional[str] = Field(default=None)
    caption: Optional[str] = Field(default=None)
    
    # Table structure
    headers: List[str] = Field(description="Column headers")
    data: List[List[Any]] = Field(description="Table data as list of rows")
    
    # Geoscientific context
    sample_ids: List[str] = Field(default_factory=list)
    location_info: Optional[GeospatialLocation] = Field(default=None)
    rock_type: Optional[RockType] = Field(default=None)
    element_data: Dict[str, List[float]] = Field(
        default_factory=dict, 
        description="Element to values mapping"
    )
    
    # Quality metrics
    data_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    extraction_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResults(BaseModel):
    """Results from literature search."""
    
    query_id: UUID
    papers: List[LiteraturePaper] = Field(default_factory=list)
    total_found: int = Field(default=0)
    search_engines_used: List[SearchEngine] = Field(default_factory=list)
    
    # Performance metrics
    search_time_seconds: Optional[float] = Field(default=None)
    relevance_scores: List[float] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractedData(BaseModel):
    """Final aggregated data from multiple papers."""
    
    id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    
    # Aggregated tables
    tables: List[DataTable] = Field(default_factory=list)
    
    # Consolidated data
    consolidated_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Consolidated data from all sources"
    )
    
    # Source tracking
    source_papers: List[UUID] = Field(default_factory=list)
    extraction_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality and validation
    data_completeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    consistency_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExecutionMetrics(BaseModel):
    """Metrics for monitoring system performance."""
    
    session_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    operation: str
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    
    # LLM usage
    tokens_used: Optional[int] = Field(default=None)
    api_calls: int = Field(default=0)
    cost_estimate: Optional[float] = Field(default=None)
    
    # Results
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the operation as complete."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        if error:
            self.error_message = error 
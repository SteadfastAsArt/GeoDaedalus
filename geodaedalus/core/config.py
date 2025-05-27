"""Configuration management for GeoDaedalus."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="structured", description="Logging format: 'structured' or 'simple'")
    output_file: Optional[Path] = Field(default=None, description="Optional log file path")
    enable_rich: bool = Field(default=True, description="Enable rich formatting")
    
    model_config = SettingsConfigDict(env_prefix="LOG_", extra="allow")


class LLMConfig(BaseSettings):
    """LLM service configuration."""
    
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Default model")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="allow")
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            v = os.getenv('OPENAI_API_KEY')
        return v


class SearchConfig(BaseSettings):
    """Search service configuration."""
    
    default_engine: str = Field(default="google", description="Default search engine")
    max_results: int = Field(default=10, description="Maximum search results")
    timeout: int = Field(default=30, description="Search timeout in seconds")
    
    # Google Search Config
    google_api_key: Optional[str] = Field(default=None, description="Google Custom Search API key")
    google_cx: Optional[str] = Field(default=None, description="Google Custom Search Engine ID")
    
    # Bing Search Config  
    bing_api_key: Optional[str] = Field(default=None, description="Bing Search API key")
    
    # SerpAPI Config
    serpapi_key: Optional[str] = Field(default=None, description="SerpAPI key")
    
    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="allow")


class ProcessingConfig(BaseSettings):
    """Document processing configuration."""
    
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    max_pdf_pages: int = Field(default=100, description="Maximum PDF pages to process")
    chunk_size: int = Field(default=1000, description="Default chunk size for text processing")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "txt", "csv", "xlsx", "json"], 
        description="Supported file formats"
    )
    
    model_config = SettingsConfigDict(env_prefix="PROCESSING_", extra="allow")


class CacheConfig(BaseSettings):
    """Cache configuration."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, description="Default TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum cache entries")
    backend: str = Field(default="memory", description="Cache backend: 'memory' or 'redis'")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for cache backend")
    
    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="allow")


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./geodaedalus.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL echo")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum overflow connections")
    
    model_config = SettingsConfigDict(env_prefix="DB_", extra="allow")


class MetricsConfig(BaseSettings):
    """Metrics collection configuration."""
    
    enabled: bool = Field(default=True, description="Enable metrics collection")
    track_execution_time: bool = Field(default=True, description="Track execution time")
    track_token_usage: bool = Field(default=True, description="Track token usage")
    track_api_costs: bool = Field(default=True, description="Track API costs")
    export_interval: int = Field(default=3600, description="Metrics export interval in seconds")
    export_format: str = Field(default="json", description="Metrics export format")
    export_path: Optional[Path] = Field(default=None, description="Path to export metrics")
    
    model_config = SettingsConfigDict(env_prefix="METRICS_", extra="allow")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(default="GeoDaedalus", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Environment
    environment: str = Field(default="development", description="Environment: development, staging, production")
    
    # Data directories
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    
    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    
    # API keys that might be passed directly
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    serpapi_key: Optional[str] = Field(default=None, description="SerpAPI Key")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Transfer top-level API keys to sub-configs if provided
        if self.openai_api_key and not self.llm.api_key:
            self.llm.api_key = self.openai_api_key
            
        if self.serpapi_key and not self.search.serpapi_key:
            self.search.serpapi_key = self.serpapi_key


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def update_settings(**kwargs) -> Settings:
    """Update settings and clear cache."""
    get_settings.cache_clear()
    return Settings(**kwargs) 
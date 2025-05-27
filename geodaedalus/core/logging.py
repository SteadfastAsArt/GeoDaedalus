"""Simplified logging configuration for GeoDaedalus using loguru only."""

import sys
from pathlib import Path
from typing import Any, Optional
from contextvars import ContextVar
from functools import wraps

from loguru import logger

from geodaedalus.core.config import LoggingConfig, get_settings


# Context variables for structured logging
_context: ContextVar[dict] = ContextVar('log_context', default={})


def configure_loguru(config: Optional[LoggingConfig] = None) -> None:
    """Configure loguru logger."""
    if config is None:
        config = get_settings().logging
    
    # Remove default logger
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level=config.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # File handler if specified
    if config.output_file:
        logger.add(
            config.output_file,
            level=config.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )


class ContextualLogger:
    """Logger with contextual information support."""
    
    def __init__(self, name: str):
        """Initialize contextual logger."""
        self.name = name
        self._logger = logger.bind(logger_name=name)
        
        # Configure logging if not already done
        if not hasattr(self.__class__, "_configured"):
            configure_loguru()
            self.__class__._configured = True
    
    def _get_context(self) -> dict:
        """Get current context."""
        return _context.get({})
    
    def _log_with_context(self, level: str, message: str, **kwargs: Any) -> None:
        """Log message with context."""
        context = self._get_context()
        all_kwargs = {**context, **kwargs}
        
        # Create message with context
        if all_kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in all_kwargs.items()])
            formatted_message = f"{message} | {context_str}"
        else:
            formatted_message = message
        
        getattr(self._logger, level)(formatted_message)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log_with_context("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log_with_context("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log_with_context("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log_with_context("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log_with_context("critical", message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        context = self._get_context()
        all_kwargs = {**context, **kwargs}
        
        if all_kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in all_kwargs.items()])
            formatted_message = f"{message} | {context_str}"
        else:
            formatted_message = message
        
        self._logger.exception(formatted_message)
    
    def bind(self, **kwargs: Any) -> "ContextualLogger":
        """Bind context to logger."""
        new_logger = ContextualLogger(self.name)
        current_context = self._get_context()
        new_context = {**current_context, **kwargs}
        _context.set(new_context)
        return new_logger


class AgentLogger(ContextualLogger):
    """Specialized logger for agent operations."""
    
    def __init__(self, agent_name: str, session_id: Optional[str] = None):
        """Initialize agent logger."""
        super().__init__(f"agent.{agent_name}")
        self.agent_name = agent_name
        self.session_id = session_id
        
        # Set agent context
        context = {"agent": agent_name}
        if session_id:
            context["session_id"] = session_id
        _context.set(context)
    
    def start_operation(self, operation: str, **context: Any) -> None:
        """Log start of operation."""
        self.info(f"Starting {operation}", operation=operation, **context)
    
    def complete_operation(self, operation: str, duration: Optional[float] = None, **context: Any) -> None:
        """Log completion of operation."""
        ctx = {"operation": operation, **context}
        if duration is not None:
            ctx["duration_seconds"] = duration
        self.info(f"Completed {operation}", **ctx)
    
    def fail_operation(self, operation: str, error: str, **context: Any) -> None:
        """Log failed operation."""
        self.error(f"Failed {operation}", operation=operation, error=error, **context)
    
    def log_llm_call(self, model: str, tokens: Optional[int] = None, cost: Optional[float] = None) -> None:
        """Log LLM API call."""
        ctx = {"model": model}
        if tokens is not None:
            ctx["tokens"] = tokens
        if cost is not None:
            ctx["cost"] = cost
        self.info("LLM API call", **ctx)
    
    def log_search_results(self, engine: str, query: str, results_count: int) -> None:
        """Log search results."""
        self.info(
            "Search completed",
            search_engine=engine,
            query=query,
            results_count=results_count
        )
    
    def log_extraction_results(self, source: str, tables_found: int, data_points: int) -> None:
        """Log data extraction results."""
        self.info(
            "Data extraction completed",
            source=source,
            tables_found=tables_found,
            data_points=data_points
        )


def get_logger(name: str) -> ContextualLogger:
    """Get a contextual logger instance."""
    return ContextualLogger(name)


def get_agent_logger(agent_name: str, session_id: Optional[str] = None) -> AgentLogger:
    """Get an agent logger instance."""
    return AgentLogger(agent_name, session_id)


def with_logging_context(**context: Any):
    """Decorator to add logging context to a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_context = _context.get({})
            new_context = {**current_context, **context}
            token = _context.set(new_context)
            try:
                return func(*args, **kwargs)
            finally:
                _context.reset(token)
        return wrapper
    return decorator 
"""Metrics and monitoring system for GeoDaedalus."""

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID, uuid4

import tiktoken
from pydantic import BaseModel

from geodaedalus.core.config import get_settings
from geodaedalus.core.logging import get_logger
from geodaedalus.core.models import ExecutionMetrics

logger = get_logger(__name__)


class TokenCounter:
    """Token counting utilities for different LLM providers."""
    
    def __init__(self):
        """Initialize token counter."""
        self._encoders: Dict[str, Any] = {}
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens for given text and model."""
        try:
            if model.startswith("gpt-"):
                encoding_name = "cl100k_base"  # For GPT-4 and GPT-3.5
            elif model.startswith("claude"):
                # Approximate for Claude (Anthropic doesn't provide exact tokenizer)
                return len(text.split()) * 1.3  # Rough approximation
            else:
                # Fallback to word-based estimation
                return len(text.split()) * 1.5
            
            if encoding_name not in self._encoders:
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            
            return len(self._encoders[encoding_name].encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to word-based estimation
            return len(text.split()) * 1.5


class MetricsCollector:
    """Collects and manages execution metrics."""
    
    def __init__(self, session_id: Optional[UUID] = None):
        """Initialize metrics collector."""
        self.session_id = session_id or uuid4()
        self.settings = get_settings()
        self.token_counter = TokenCounter()
        self._metrics: List[ExecutionMetrics] = []
        self._active_operations: Dict[str, ExecutionMetrics] = {}
    
    @contextmanager
    def track_operation(
        self,
        agent_name: str,
        operation: str,
        **metadata: Any
    ) -> Iterator[ExecutionMetrics]:
        """Context manager for tracking operation metrics."""
        metric = ExecutionMetrics(
            session_id=self.session_id,
            agent_name=agent_name,
            operation=operation,
            start_time=datetime.utcnow(),
            metadata=metadata
        )
        
        operation_key = f"{agent_name}:{operation}:{time.time()}"
        self._active_operations[operation_key] = metric
        
        try:
            yield metric
            metric.mark_complete(success=True)
        except Exception as e:
            metric.mark_complete(success=False, error=str(e))
            raise
        finally:
            self._metrics.append(metric)
            self._active_operations.pop(operation_key, None)
            
            if self.settings.metrics.track_execution_time:
                logger.info(
                    f"Operation completed: {operation}",
                    agent=agent_name,
                    duration=metric.duration_seconds,
                    success=metric.success
                )
    
    def track_llm_call(
        self,
        agent_name: str,
        model: str,
        prompt: str,
        response: str,
        cost: Optional[float] = None
    ) -> None:
        """Track LLM API call metrics."""
        if not self.settings.metrics.track_token_usage:
            return
        
        prompt_tokens = self.token_counter.count_tokens(prompt, model)
        response_tokens = self.token_counter.count_tokens(response, model)
        total_tokens = prompt_tokens + response_tokens
        
        metric = ExecutionMetrics(
            session_id=self.session_id,
            agent_name=agent_name,
            operation="llm_call",
            start_time=datetime.utcnow(),
            tokens_used=total_tokens,
            api_calls=1,
            cost_estimate=cost,
            metadata={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }
        )
        metric.mark_complete(success=True)
        self._metrics.append(metric)
        
        logger.info(
            "LLM call tracked",
            agent=agent_name,
            model=model,
            tokens=total_tokens,
            cost=cost
        )
    
    def track_api_call(
        self,
        agent_name: str,
        api_name: str,
        endpoint: str,
        success: bool,
        response_time: Optional[float] = None,
        **metadata: Any
    ) -> None:
        """Track general API call metrics."""
        if not self.settings.metrics.track_api_calls:
            return
        
        metric = ExecutionMetrics(
            session_id=self.session_id,
            agent_name=agent_name,
            operation=f"api_call_{api_name}",
            start_time=datetime.utcnow(),
            api_calls=1,
            metadata={
                "api_name": api_name,
                "endpoint": endpoint,
                "response_time": response_time,
                **metadata
            }
        )
        
        if response_time:
            metric.duration_seconds = response_time
        
        metric.mark_complete(success=success)
        self._metrics.append(metric)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of metrics for current session."""
        if not self._metrics:
            return {"session_id": str(self.session_id), "total_operations": 0}
        
        total_operations = len(self._metrics)
        successful_operations = sum(1 for m in self._metrics if m.success)
        total_duration = sum(m.duration_seconds or 0 for m in self._metrics)
        total_tokens = sum(m.tokens_used or 0 for m in self._metrics)
        total_api_calls = sum(m.api_calls for m in self._metrics)
        total_cost = sum(m.cost_estimate or 0 for m in self._metrics)
        
        # Group by agent
        agent_stats = {}
        for metric in self._metrics:
            agent = metric.agent_name
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "operations": 0,
                    "success_rate": 0,
                    "total_duration": 0,
                    "tokens_used": 0,
                    "api_calls": 0,
                    "cost": 0
                }
            
            stats = agent_stats[agent]
            stats["operations"] += 1
            stats["total_duration"] += metric.duration_seconds or 0
            stats["tokens_used"] += metric.tokens_used or 0
            stats["api_calls"] += metric.api_calls
            stats["cost"] += metric.cost_estimate or 0
        
        # Calculate success rates
        for agent, stats in agent_stats.items():
            agent_metrics = [m for m in self._metrics if m.agent_name == agent]
            successful_count = sum(1 for m in agent_metrics if m.success)
            stats["success_rate"] = successful_count / len(agent_metrics) if agent_metrics else 0
        
        return {
            "session_id": str(self.session_id),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": successful_operations / total_operations,
            "total_duration_seconds": total_duration,
            "total_tokens_used": total_tokens,
            "total_api_calls": total_api_calls,
            "total_cost_estimate": total_cost,
            "agent_statistics": agent_stats,
            "start_time": min(m.start_time for m in self._metrics).isoformat(),
            "end_time": max(m.end_time for m in self._metrics if m.end_time).isoformat()
        }
    
    def export_metrics(self, output_path: Optional[Path] = None) -> Path:
        """Export metrics to JSON file."""
        if output_path is None:
            output_path = (
                self.settings.metrics.metrics_output_dir / 
                f"metrics_{self.session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to serializable format
        serializable_metrics = []
        for metric in self._metrics:
            metric_dict = metric.dict()
            # Convert datetime objects to ISO strings
            for key, value in metric_dict.items():
                if isinstance(value, datetime):
                    metric_dict[key] = value.isoformat()
                elif isinstance(value, UUID):
                    metric_dict[key] = str(value)
            serializable_metrics.append(metric_dict)
        
        data = {
            "session_summary": self.get_session_summary(),
            "detailed_metrics": serializable_metrics
        }
        
        with output_path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {output_path}")
        return output_path
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._active_operations.clear()


class CostEstimator:
    """Estimates costs for different LLM providers."""
    
    # Pricing per 1K tokens (approximate, as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        response_tokens: int
    ) -> Optional[float]:
        """Estimate cost for LLM call."""
        if model not in self.PRICING:
            return None
        
        pricing = self.PRICING[model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (response_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def reset_metrics_collector() -> MetricsCollector:
    """Reset global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector 
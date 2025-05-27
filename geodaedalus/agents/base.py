"""Base agent class for GeoDaedalus multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from uuid import UUID, uuid4

from geodaedalus.core.config import get_settings, Settings
from geodaedalus.core.logging import get_agent_logger, AgentLogger
from geodaedalus.core.metrics import get_metrics_collector, MetricsCollector
from geodaedalus.core.models import ExecutionMetrics

T = TypeVar('T')
U = TypeVar('U')


class BaseAgent(ABC, Generic[T, U]):
    """Base class for all GeoDaedalus agents."""
    
    def __init__(
        self,
        agent_name: str,
        session_id: Optional[UUID] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize base agent."""
        self.agent_name = agent_name
        self.session_id = session_id or uuid4()
        self.settings = settings or get_settings()
        
        # Initialize logging and metrics
        self.logger = get_agent_logger(agent_name, str(self.session_id))
        self.metrics = get_metrics_collector()
        
        self.logger.info(f"Initialized {agent_name} agent", session_id=str(self.session_id))
    
    @abstractmethod
    async def process(self, input_data: T, **kwargs: Any) -> U:
        """Process input data and return results.
        
        Args:
            input_data: Input data of type T
            **kwargs: Additional arguments
            
        Returns:
            Processed results of type U
        """
        pass
    
    async def execute_with_metrics(
        self,
        operation_name: str,
        input_data: T,
        **kwargs: Any
    ) -> U:
        """Execute agent processing with metrics tracking."""
        with self.metrics.track_operation(
            agent_name=self.agent_name,
            operation=operation_name,
            input_type=type(input_data).__name__,
            **kwargs
        ) as metric:
            self.logger.start_operation(operation_name, input_type=type(input_data).__name__)
            
            try:
                result = await self.process(input_data, **kwargs)
                
                self.logger.complete_operation(
                    operation_name,
                    duration=metric.duration_seconds,
                    result_type=type(result).__name__
                )
                
                return result
                
            except Exception as e:
                self.logger.fail_operation(operation_name, str(e))
                raise
    
    def validate_input(self, input_data: T) -> bool:
        """Validate input data format and content.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return input_data is not None
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and status."""
        return {
            "name": self.agent_name,
            "session_id": str(self.session_id),
            "type": self.__class__.__name__,
            "settings": {
                "llm_provider": self.settings.llm.provider,
                "llm_model": self.settings.llm.model,
            }
        } 
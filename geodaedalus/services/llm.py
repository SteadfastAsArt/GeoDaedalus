"""LLM service for GeoDaedalus agents."""

import asyncio
from typing import Any, Dict, Optional, Union

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from geodaedalus.core.config import Settings, LLMConfig
from geodaedalus.core.logging import get_logger
from geodaedalus.core.metrics import get_metrics_collector, CostEstimator

logger = get_logger(__name__)


class LLMResponse:
    """Response from LLM service."""
    
    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_estimate: Optional[float] = None,
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost_estimate = cost_estimate


class LLMService:
    """Service for LLM interactions with multiple provider support."""
    
    def __init__(self, settings: Settings):
        """Initialize LLM service."""
        self.settings = settings
        self.config = settings.llm
        self.metrics = get_metrics_collector()
        self.cost_estimator = CostEstimator()
        
        # Initialize providers
        self._clients: Dict[str, Any] = {}
        self._init_providers()
    
    def _init_providers(self) -> None:
        """Initialize LLM provider clients."""
        # OpenAI
        if self.settings.openai_api_key:
            try:
                import openai
                self._clients["openai"] = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not available")
        
        # Anthropic
        if self.settings.anthropic_api_key:
            try:
                import anthropic
                self._clients["anthropic"] = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not available")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, ConnectionError))
    )
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate response from LLM."""
        
        # Use defaults from config if not specified
        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Determine provider from model name
        provider = self._get_provider_from_model(model)
        
        if provider not in self._clients:
            raise ValueError(f"Provider {provider} not available or not configured")
        
        logger.info(
            "Generating LLM response",
            provider=provider,
            model=model,
            prompt_length=len(prompt),
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        try:
            if provider == "openai":
                response = await self._call_openai(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif provider == "anthropic":
                response = await self._call_anthropic(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Track metrics
            if agent_name:
                self.metrics.track_llm_call(
                    agent_name=agent_name,
                    model=model,
                    prompt=prompt,
                    response=response.content,
                    cost=response.cost_estimate
                )
            
            logger.info(
                "LLM response generated successfully",
                provider=provider,
                model=model,
                response_length=len(response.content),
                tokens_used=response.total_tokens,
                cost=response.cost_estimate
            )
            
            return response.content
            
        except Exception as e:
            logger.error(
                f"LLM generation failed: {e}",
                provider=provider,
                model=model,
                error=str(e)
            )
            raise
    
    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any
    ) -> LLMResponse:
        """Call OpenAI API."""
        client = self._clients["openai"]
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        usage = response.usage
        content = response.choices[0].message.content
        
        # Estimate cost
        cost = None
        if usage and usage.prompt_tokens and usage.completion_tokens:
            cost = self.cost_estimator.estimate_cost(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                response_tokens=usage.completion_tokens
            )
        
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cost_estimate=cost
        )
    
    async def _call_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any
    ) -> LLMResponse:
        """Call Anthropic API."""
        client = self._clients["anthropic"]
        
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        content = response.content[0].text
        
        # Estimate tokens (Anthropic doesn't provide exact counts)
        from geodaedalus.core.metrics import TokenCounter
        token_counter = TokenCounter()
        prompt_tokens = token_counter.count_tokens(prompt, model)
        completion_tokens = token_counter.count_tokens(content, model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Estimate cost
        cost = self.cost_estimator.estimate_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            response_tokens=completion_tokens
        )
        
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost
        )
    
    def _get_provider_from_model(self, model: str) -> str:
        """Determine provider from model name."""
        if model.startswith("gpt-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        else:
            # Default to configured provider
            return self.config.provider
    
    async def batch_generate(
        self,
        prompts: list[str],
        model: Optional[str] = None,
        concurrency_limit: int = 5,
        agent_name: Optional[str] = None,
        **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts with concurrency control."""
        
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def _generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate_response(
                    prompt=prompt,
                    model=model,
                    agent_name=agent_name,
                    **kwargs
                )
        
        tasks = [_generate_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to generate response for prompt {i}: {response}")
                results.append("")  # Empty string for failed responses
            else:
                results.append(response)
        
        return results 
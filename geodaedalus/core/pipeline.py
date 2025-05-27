"""Pipeline orchestrator for GeoDaedalus multi-agent system."""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from geodaedalus.agents import (
    RequirementUnderstandingAgent,
    LiteratureSearchAgent,
    DataExtractionAgent,
    DataFusionAgent,
)
from geodaedalus.core.config import get_settings, Settings
from geodaedalus.core.logging import get_agent_logger
from geodaedalus.core.metrics import get_metrics_collector
from geodaedalus.core.models import (
    GeoscientificConstraints,
    SearchResults,
    ExtractedData,
    SearchEngine,
    QueryRequest,
)


class GeoDaedalusPipeline:
    """Main pipeline orchestrator for the GeoDaedalus system."""
    
    def __init__(
        self,
        session_id: Optional[UUID] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize the pipeline."""
        self.session_id = session_id or uuid4()
        self.settings = settings or get_settings()
        
        # Initialize logging and metrics
        self.logger = get_agent_logger("pipeline", str(self.session_id))
        self.metrics = get_metrics_collector()
        
        # Initialize agents
        self.agent1 = RequirementUnderstandingAgent(
            session_id=self.session_id,
            settings=self.settings
        )
        self.agent2 = LiteratureSearchAgent(
            session_id=self.session_id,
            settings=self.settings
        )
        self.agent3 = DataExtractionAgent(
            session_id=self.session_id,
            settings=self.settings
        )
        self.agent4 = DataFusionAgent(
            session_id=self.session_id,
            settings=self.settings
        )
        
        self.logger.info("GeoDaedalus pipeline initialized", session_id=str(self.session_id))
    
    async def process_query(
        self,
        user_query: str,
        max_papers: int = 20,
        max_extraction_papers: Optional[int] = None,
        search_engines: Optional[List[SearchEngine]] = None,
        validation_level: str = "standard",
        **kwargs: Any
    ) -> ExtractedData:
        """Process a complete user query through the entire pipeline.
        
        Args:
            user_query: Natural language query from user
            max_papers: Maximum number of papers to search for
            max_extraction_papers: Maximum papers to extract data from (defaults to max_papers)
            search_engines: List of search engines to use
            validation_level: Data validation level ("lenient", "standard", "strict")
            **kwargs: Additional arguments
            
        Returns:
            Final extracted and validated data
        """
        query_request = QueryRequest(
            user_query=user_query,
            max_results=max_papers,
            search_engines=search_engines or [SearchEngine.SEMANTIC_SCHOLAR, SearchEngine.GOOGLE_SCHOLAR]
        )
        
        max_extraction_papers = max_extraction_papers or max_papers
        
        self.logger.info(
            "Starting complete pipeline processing",
            query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
            max_papers=max_papers,
            max_extraction_papers=max_extraction_papers,
            search_engines=query_request.search_engines,
            validation_level=validation_level
        )
        
        with self.metrics.track_operation(
            agent_name="pipeline",
            operation="complete_processing",
            input_type="user_query",
            query_id=str(query_request.id)
        ) as metric:
            try:
                # Step 1: Requirement Understanding
                self.logger.info("Step 1: Understanding requirements")
                constraints = await self.agent1.execute_with_metrics(
                    "process_query",
                    user_query,
                    query_id=query_request.id
                )
                
                # Step 2: Literature Search
                self.logger.info("Step 2: Searching literature")
                search_results = await self.agent2.execute_with_metrics(
                    "search_literature",
                    constraints,
                    max_results=max_papers,
                    search_engines=query_request.search_engines,
                    query_id=query_request.id
                )
                
                # Step 3: Data Extraction
                self.logger.info("Step 3: Extracting data")
                extracted_data = await self.agent3.execute_with_metrics(
                    "extract_data",
                    search_results,
                    max_papers=max_extraction_papers,
                    query_id=query_request.id
                )
                
                # Step 4: Data Fusion and Validation
                self.logger.info("Step 4: Fusing and validating data")
                final_data = await self.agent4.execute_with_metrics(
                    "fuse_data",
                    extracted_data,
                    validation_level=validation_level,
                    query_id=query_request.id
                )
                
                # Add pipeline metadata
                final_data.consolidated_data.update({
                    "pipeline_metadata": {
                        "session_id": str(self.session_id),
                        "original_query": user_query,
                        "constraints_extracted": constraints.dict(),
                        "search_summary": {
                            "engines_used": search_results.search_engines_used,
                            "papers_found": len(search_results.papers),
                            "search_time": search_results.search_time_seconds
                        },
                        "processing_summary": {
                            "max_papers_requested": max_papers,
                            "papers_processed": len(extracted_data.source_papers),
                            "tables_extracted": len(extracted_data.tables),
                            "final_tables": len(final_data.tables)
                        }
                    }
                })
                
                self.logger.info(
                    "Pipeline processing completed successfully",
                    papers_found=len(search_results.papers),
                    papers_processed=len(extracted_data.source_papers),
                    tables_extracted=len(extracted_data.tables),
                    final_tables=len(final_data.tables),
                    data_completeness=final_data.data_completeness,
                    consistency_score=final_data.consistency_score
                )
                
                return final_data
                
            except Exception as e:
                self.logger.error("Pipeline processing failed", error=str(e))
                raise
    
    async def process_step_by_step(
        self,
        user_query: str,
        max_papers: int = 20,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Process query step by step, returning intermediate results.
        
        Useful for debugging and understanding the pipeline flow.
        """
        results = {}
        
        try:
            # Step 1: Requirement Understanding
            self.logger.info("Step 1: Understanding requirements")
            constraints = await self.agent1.execute_with_metrics(
                "process_query",
                user_query
            )
            results["step1_constraints"] = constraints
            
            # Step 2: Literature Search
            self.logger.info("Step 2: Searching literature")
            search_results = await self.agent2.execute_with_metrics(
                "search_literature",
                constraints,
                max_results=max_papers
            )
            results["step2_search"] = search_results
            
            # Step 3: Data Extraction (limit to first 5 papers for debugging)
            self.logger.info("Step 3: Extracting data")
            extracted_data = await self.agent3.execute_with_metrics(
                "extract_data",
                search_results,
                max_papers=min(5, len(search_results.papers))
            )
            results["step3_extraction"] = extracted_data
            
            # Step 4: Data Fusion and Validation
            self.logger.info("Step 4: Fusing and validating data")
            final_data = await self.agent4.execute_with_metrics(
                "fuse_data",
                extracted_data
            )
            results["step4_fusion"] = final_data
            
            return results
            
        except Exception as e:
            self.logger.error("Step-by-step processing failed", error=str(e))
            results["error"] = str(e)
            return results
    
    async def validate_constraints(self, user_query: str) -> GeoscientificConstraints:
        """Validate and extract constraints from user query only."""
        return await self.agent1.execute_with_metrics(
            "process_query",
            user_query
        )
    
    async def search_literature_only(
        self,
        constraints: GeoscientificConstraints,
        max_results: int = 50,
        search_engines: Optional[List[SearchEngine]] = None
    ) -> SearchResults:
        """Perform literature search only."""
        return await self.agent2.execute_with_metrics(
            "search_literature",
            constraints,
            max_results=max_results,
            search_engines=search_engines
        )
    
    async def extract_data_only(
        self,
        search_results: SearchResults,
        max_papers: Optional[int] = None
    ) -> ExtractedData:
        """Perform data extraction only."""
        return await self.agent3.execute_with_metrics(
            "extract_data",
            search_results,
            max_papers=max_papers
        )
    
    async def fuse_data_only(
        self,
        extracted_data: ExtractedData,
        validation_level: str = "standard"
    ) -> ExtractedData:
        """Perform data fusion and validation only."""
        return await self.agent4.execute_with_metrics(
            "fuse_data",
            extracted_data,
            validation_level=validation_level
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        return {
            "session_id": str(self.session_id),
            "agents": {
                "agent1": self.agent1.get_agent_info(),
                "agent2": self.agent2.get_agent_info(),
                "agent3": self.agent3.get_agent_info(),
                "agent4": self.agent4.get_agent_info(),
            },
            "settings": {
                "llm_provider": self.settings.llm.provider,
                "llm_model": self.settings.llm.model,
                "max_tokens": self.settings.llm.max_tokens,
                "temperature": self.settings.llm.temperature,
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the pipeline and cleanup resources."""
        self.logger.info("Closing GeoDaedalus pipeline")
        
        # Close all agents
        if hasattr(self.agent1, 'close'):
            await self.agent1.close()
        if hasattr(self.agent2, 'close'):
            await self.agent2.close()
        if hasattr(self.agent3, 'close'):
            await self.agent3.close()
        if hasattr(self.agent4, 'close'):
            await self.agent4.close()
        
        self.logger.info("Pipeline closed successfully") 
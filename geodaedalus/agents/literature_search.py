"""Agent 2: Literature Search - Searches academic databases with LLM relevance filtering."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from geodaedalus.agents.base import BaseAgent
from geodaedalus.core.models import (
    GeoscientificConstraints,
    LiteraturePaper,
    SearchResults,
    SearchEngine,
    Author,
)
from geodaedalus.services.llm import LLMService


class LiteratureSearchAgent(BaseAgent[GeoscientificConstraints, SearchResults]):
    """Agent for searching academic literature with relevance filtering."""

    def __init__(self, **kwargs):
        """Initialize literature search agent."""
        super().__init__("literature_search", **kwargs)
        self.llm_service = LLMService(self.settings)
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Search engine configurations
        self.search_configs = {
            SearchEngine.SEMANTIC_SCHOLAR: {
                "base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "rate_limit": 100,  # requests per minute
            },
            SearchEngine.GOOGLE_SCHOLAR: {
                "base_url": "https://serpapi.com/search.json",
                "rate_limit": 100,
            },
        }

    async def process(
            self,
            constraints: GeoscientificConstraints,
            max_results: int = 50,
            search_engines: Optional[List[SearchEngine]] = None,
            **kwargs: Any
    ) -> SearchResults:
        """Search literature based on geoscientific constraints."""
        if not self.validate_input(constraints):
            raise ValueError("Invalid constraints provided")

        search_engines = search_engines or [
            SearchEngine.SEMANTIC_SCHOLAR,
            SearchEngine.GOOGLE_SCHOLAR
        ]

        self.logger.info(
            "Starting literature search",
            search_engines=search_engines,
            max_results=max_results,
            rock_types=len(constraints.rock_types),
            element_constraints=len(constraints.element_constraints)
        )

        # Generate search queries from constraints
        search_queries = await self._generate_search_queries(constraints)

        # Search across multiple engines
        all_papers = []
        for engine in search_engines:
            try:
                papers = await self._search_engine(engine, search_queries, max_results // len(search_engines))
                all_papers.extend(papers)
                self.logger.info(f"Found {len(papers)} papers from {engine.value}")
            except Exception as e:
                self.logger.error(f"Search failed for {engine.value}", error=str(e))

        # Remove duplicates
        unique_papers = await self._deduplicate_papers(all_papers)

        # Filter for relevance using LLM
        relevant_papers = await self._filter_relevance(unique_papers, constraints)

        # Sort by relevance score
        relevant_papers.sort(key=lambda p: p.relevance_score or 0, reverse=True)

        # Limit results
        final_papers = relevant_papers[:max_results]

        self.logger.info(
            "Literature search completed",
            total_found=len(all_papers),
            unique_papers=len(unique_papers),
            relevant_papers=len(relevant_papers),
            final_results=len(final_papers)
        )

        return SearchResults(
            query_id=kwargs.get('query_id', self.session_id),
            papers=final_papers,
            total_found=len(all_papers),
            search_engines_used=search_engines,
            relevance_scores=[p.relevance_score or 0 for p in final_papers]
        )

    async def _generate_search_queries(self, constraints: GeoscientificConstraints) -> List[str]:
        """Generate search queries from constraints using LLM."""
        # TODO: check prompt
        prompt = f"""
        Generate 3-5 academic search queries for finding geoscience literature based on these constraints:
        
        Rock Types: {[rt.value for rt in constraints.rock_types]}
        Element Constraints: {[f"{ec.category.value}: {ec.elements}" for ec in constraints.element_constraints]}
        Location: {constraints.spatial.location_name if constraints.spatial else "Not specified"}
        Geological Period: {constraints.temporal.geological_period if constraints.temporal else "Not specified"}
        Additional Keywords: {constraints.additional_keywords}
        
        Generate queries that would find relevant geochemical and geological papers.
        Return as a JSON list of strings.
        """

        # TODO: check result, should be a string, JSON format
        response = await self.llm_service.generate_response(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )

        try:
            queries = json.loads(response)
            if isinstance(queries, list):
                return queries
        except json.JSONDecodeError:
            pass

        # Fallback: generate basic queries from keywords
        keywords = constraints.to_search_keywords()
        return [
            " ".join(keywords[:3]),
            " ".join(keywords[1:4]) if len(keywords) > 3 else " ".join(keywords),
            f"geochemistry {' '.join(keywords[:2])}" if keywords else "geochemistry"
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _search_engine(
            self,
            engine: SearchEngine,
            queries: List[str],
            max_results: int
    ) -> List[LiteraturePaper]:
        """Search a specific academic search engine."""
        # TODO: add search engine info
        if engine == SearchEngine.SEMANTIC_SCHOLAR:
            return await self._search_semantic_scholar(queries, max_results)
        elif engine == SearchEngine.GOOGLE_SCHOLAR:
            return await self._search_google_scholar(queries, max_results)
        else:
            self.logger.warning(f"Search engine {engine.value} not implemented")
            return []

    async def _search_semantic_scholar(self, queries: List[str], max_results: int) -> List[LiteraturePaper]:
        """Search Semantic Scholar API."""
        papers = []

        for query in queries:
            # ref: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_relevance_search
            try:
                params = {
                    "query": query,
                    "limit": min(max_results // len(queries), 100),
                    "fields": "paperId,title,authors,abstract,year,journal,citationCount,isOpenAccess,url,venue"
                }

                response = await self.http_client.get(
                    self.search_configs[SearchEngine.SEMANTIC_SCHOLAR]["base_url"],
                    params=params
                )
                response.raise_for_status()

                data = response.json()

                for paper_data in data.get("data", []):
                    paper = self._parse_semantic_scholar_paper(paper_data)
                    if paper:
                        papers.append(paper)

                # Rate limiting
                await asyncio.sleep(0.6)  # 100 requests per minute

            except Exception as e:
                self.logger.error(f"Semantic Scholar search failed for query '{query}'", error=str(e))

        return papers

    async def _search_google_scholar(self, queries: List[str], max_results: int) -> List[LiteraturePaper]:
        """Search Google Scholar via SerpAPI."""
        if not self.settings.search.serpapi_key:
            self.logger.warning("SerpAPI key not configured, skipping Google Scholar search")
            return []

        papers = []

        for query in queries:
            # ref: https://serpapi.com/google-scholar-api
            try:
                params = {
                    "engine": "google_scholar",
                    "q": query,
                    "api_key": self.settings.search.serpapi_key,
                    "num": min(max_results // len(queries), 20),
                    "as_ylo": 2000,  # Papers from 2000 onwards
                }

                response = await self.http_client.get(
                    self.search_configs[SearchEngine.GOOGLE_SCHOLAR]["base_url"],
                    params=params
                )
                response.raise_for_status()

                data = response.json()

                for result in data.get("organic_results", []):
                    paper = self._parse_google_scholar_paper(result)
                    if paper:
                        papers.append(paper)

                # Rate limiting
                await asyncio.sleep(0.6)

            except Exception as e:
                self.logger.error(f"Google Scholar search failed for query '{query}'", error=str(e))

        return papers

    def _parse_semantic_scholar_paper(self, data: Dict[str, Any]) -> Optional[LiteraturePaper]:
        """Parse Semantic Scholar paper data."""
        try:
            # TODO: add other fields like keywords, etc.
            authors = []
            for author_data in data.get("authors", []):
                authors.append(Author(name=author_data.get("name", "Unknown")))

            pdf_url = None
            if (data.get("isOpenAccess", False)
                    and data.get("openAccessPdf", False)
                    and data.get("openAccessPdf", {}).get("url", False)):
                pdf_url = data.get("openAccessPdf", {}).get("url", False)

            return LiteraturePaper(
                title=data.get("title", ""),
                authors=authors,
                abstract=data.get("abstract"),
                journal=data.get("venue") or data.get("journal"),
                year=data.get("year"),
                citation_count=data.get("citationCount"),
                is_open_access=data.get("isOpenAccess", False),
                pdf_url=pdf_url,
                web_url=data.get("url"),
                keywords=[]
            )
        except Exception as e:
            self.logger.error("Failed to parse Semantic Scholar paper", error=str(e))
            return None

    def _parse_google_scholar_paper(self, data: Dict[str, Any]) -> Optional[LiteraturePaper]:
        """Parse Google Scholar paper data."""
        try:
            # ref: https://serpapi.com/google-scholar-organic-results

            # Extract authors from publication info
            authors = []
            publication_info = data.get("publication_info", {})
            authors_list = publication_info.get("authors", [])
            if len(authors_list) > 0:
                author_names = [au.get("name") for au in authors_list]
                authors = [Author(name=name) for name in author_names if name]

            # TODO: Extract pdf_url from resources
            pdf_url = None
            resources = data.get("resources", [])
            if len(resources) > 0:
                resource = resources[0]
                if resource.get("file_format", None) == "PDF":
                    pdf_url = resource.get("link", None)

            return LiteraturePaper(
                title=data.get("title", ""),
                authors=authors,
                abstract=data.get("snippet"),
                journal=publication_info.get("summary", "").split(",")[0] if publication_info.get("summary") else None,
                pdf_url=pdf_url,
                web_url=data.get("link"),
                citation_count=data.get("inline_links", {}).get("cited_by", {}).get("total"),
                keywords=[]
            )
        except Exception as e:
            self.logger.error("Failed to parse Google Scholar paper", error=str(e))
            return None

    async def _deduplicate_papers(self, papers: List[LiteraturePaper]) -> List[LiteraturePaper]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()

            # Check for exact matches
            if normalized_title in seen_titles:
                continue

            # Check for similar titles (simple approach)
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)

        return unique_papers

    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar using simple word overlap."""
        words1 = set(title1.split())
        words2 = set(title2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union >= threshold

    async def _filter_relevance(
            self,
            papers: List[LiteraturePaper],
            constraints: GeoscientificConstraints
    ) -> List[LiteraturePaper]:
        """Filter papers for relevance using LLM."""
        if not papers:
            return []

        # Process papers in batches to avoid token limits
        batch_size = 5
        relevant_papers = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_results = await self._score_paper_batch(batch, constraints)
            relevant_papers.extend(batch_results)

        # Filter papers with relevance score > 0.3
        return [p for p in relevant_papers if (p.relevance_score or 0) > 0.3]

    async def _score_paper_batch(
            self,
            papers: List[LiteraturePaper],
            constraints: GeoscientificConstraints
    ) -> List[LiteraturePaper]:
        """Score a batch of papers for relevance."""
        papers_info = []
        for i, paper in enumerate(papers):
            papers_info.append({
                "index": i,
                "title": paper.title,
                "abstract": paper.abstract or "No abstract available",
                "journal": paper.journal or "Unknown journal",
                "year": paper.year
            })

        prompt = f"""
        Rate the relevance of these papers to the following geoscientific research constraints:
        
        Target Rock Types: {[rt.value for rt in constraints.rock_types]}
        Target Elements: {[f"{ec.category.value}: {ec.elements}" for ec in constraints.element_constraints]}
        Location: {constraints.spatial.location_name if constraints.spatial else "Any location"}
        Time Period: {constraints.temporal.geological_period if constraints.temporal else "Any period"}
        
        Papers to evaluate:
        {json.dumps(papers_info, indent=2)}
        
        For each paper, provide a relevance score from 0.0 to 1.0 where:
        - 1.0 = Highly relevant (directly addresses the research constraints)
        - 0.7 = Moderately relevant (related but not directly addressing constraints)
        - 0.4 = Somewhat relevant (tangentially related)
        - 0.0 = Not relevant
        
        Return as JSON: {{"scores": [0.8, 0.3, 0.9, ...]}}
        """
        # TODO: add log

        try:
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )

            result = json.loads(response.content)
            scores = result.get("scores", [])

            # Apply scores to papers
            for i, score in enumerate(scores):
                if i < len(papers):
                    papers[i].relevance_score = float(score)

        except Exception as e:
            self.logger.error("Failed to score paper relevance", error=str(e))
            # Fallback: assign default scores
            for paper in papers:
                paper.relevance_score = 0.5

        return papers

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()

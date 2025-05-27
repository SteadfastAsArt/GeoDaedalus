"""Academic search service for GeoDaedalus."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from geodaedalus.core.config import Settings
from geodaedalus.core.logging import get_logger
from geodaedalus.core.models import LiteraturePaper, SearchEngine, Author

logger = get_logger(__name__)


class BaseSearchEngine(ABC):
    """Base class for academic search engines."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        **kwargs: Any
    ) -> List[LiteraturePaper]:
        """Search for papers."""
        pass
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()


class SemanticScholarEngine(BaseSearchEngine):
    """Semantic Scholar search engine."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        **kwargs: Any
    ) -> List[LiteraturePaper]:
        """Search Semantic Scholar."""
        papers = []
        
        try:
            params = {
                "query": query,
                "limit": min(max_results, 100),
                "fields": "paperId,title,authors,abstract,year,journal,citationCount,isOpenAccess,url,venue,externalIds"
            }
            
            response = await self.http_client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for paper_data in data.get("data", []):
                paper = self._parse_paper(paper_data)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Semantic Scholar found {len(papers)} papers for query: {query}")
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
        
        return papers
    
    def _parse_paper(self, data: Dict[str, Any]) -> Optional[LiteraturePaper]:
        """Parse Semantic Scholar paper data."""
        try:
            authors = []
            for author_data in data.get("authors", []):
                authors.append(Author(
                    name=author_data.get("name", "Unknown"),
                    affiliation=author_data.get("affiliations", [{}])[0].get("name") if author_data.get("affiliations") else None
                ))
            
            # Extract DOI from external IDs
            doi = None
            external_ids = data.get("externalIds", {})
            if external_ids and "DOI" in external_ids:
                doi = external_ids["DOI"]
            
            return LiteraturePaper(
                title=data.get("title", ""),
                authors=authors,
                abstract=data.get("abstract"),
                journal=data.get("venue") or data.get("journal"),
                year=data.get("year"),
                doi=doi,
                citation_count=data.get("citationCount"),
                is_open_access=data.get("isOpenAccess", False),
                web_url=data.get("url"),
                keywords=[]
            )
        except Exception as e:
            logger.error(f"Failed to parse Semantic Scholar paper: {e}")
            return None


class GoogleScholarEngine(BaseSearchEngine):
    """Google Scholar search engine via SerpAPI."""
    
    BASE_URL = "https://serpapi.com/search.json"
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.api_key = settings.serpapi_key
        if not self.api_key:
            logger.warning("SerpAPI key not configured - Google Scholar search unavailable")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        **kwargs: Any
    ) -> List[LiteraturePaper]:
        """Search Google Scholar via SerpAPI."""
        if not self.api_key:
            logger.warning("SerpAPI key not available, skipping Google Scholar search")
            return []
        
        papers = []
        
        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 20),
                "as_ylo": 2000,  # Papers from 2000 onwards
                "hl": "en",
            }
            
            response = await self.http_client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for result in data.get("organic_results", []):
                paper = self._parse_paper(result)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Google Scholar found {len(papers)} papers for query: {query}")
            
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
        
        return papers
    
    def _parse_paper(self, data: Dict[str, Any]) -> Optional[LiteraturePaper]:
        """Parse Google Scholar paper data."""
        try:
            # Extract authors from publication info
            authors = []
            publication_info = data.get("publication_info", {})
            authors_str = publication_info.get("authors", "")
            if authors_str:
                author_names = [name.strip() for name in authors_str.split(",")]
                authors = [Author(name=name) for name in author_names if name]
            
            # Extract year from publication info
            year = None
            summary = publication_info.get("summary", "")
            if summary:
                # Try to extract year from summary (e.g., "Journal Name, 2023")
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', summary)
                if year_match:
                    year = int(year_match.group())
            
            # Extract journal name
            journal = None
            if summary:
                # Take first part before comma as journal name
                journal = summary.split(",")[0].strip()
            
            return LiteraturePaper(
                title=data.get("title", ""),
                authors=authors,
                abstract=data.get("snippet"),
                journal=journal,
                year=year,
                web_url=data.get("link"),
                pdf_url=data.get("resources", [{}])[0].get("link") if data.get("resources") else None,
                citation_count=data.get("inline_links", {}).get("cited_by", {}).get("total"),
                keywords=[]
            )
        except Exception as e:
            logger.error(f"Failed to parse Google Scholar paper: {e}")
            return None


class CrossRefEngine(BaseSearchEngine):
    """CrossRef search engine for DOI-based search."""
    
    BASE_URL = "https://api.crossref.org/works"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        **kwargs: Any
    ) -> List[LiteraturePaper]:
        """Search CrossRef."""
        papers = []
        
        try:
            params = {
                "query": query,
                "rows": min(max_results, 1000),
                "sort": "relevance",
                "filter": "has-abstract:true"
            }
            
            headers = {
                "User-Agent": "GeoDaedalus/1.0 (mailto:geodaedalus@example.org)"
            }
            
            response = await self.http_client.get(
                self.BASE_URL, 
                params=params, 
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get("message", {}).get("items", []):
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            logger.info(f"CrossRef found {len(papers)} papers for query: {query}")
            
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
        
        return papers
    
    def _parse_paper(self, data: Dict[str, Any]) -> Optional[LiteraturePaper]:
        """Parse CrossRef paper data."""
        try:
            # Extract authors
            authors = []
            for author_data in data.get("author", []):
                given = author_data.get("given", "")
                family = author_data.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(Author(
                        name=name,
                        affiliation=author_data.get("affiliation", [{}])[0].get("name") if author_data.get("affiliation") else None
                    ))
            
            # Extract publication date
            year = None
            date_parts = data.get("published-print", {}).get("date-parts") or data.get("published-online", {}).get("date-parts")
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            
            # Extract journal
            journal = None
            container_title = data.get("container-title", [])
            if container_title:
                journal = container_title[0]
            
            return LiteraturePaper(
                title=" ".join(data.get("title", [])),
                authors=authors,
                abstract=data.get("abstract"),
                journal=journal,
                year=year,
                doi=data.get("DOI"),
                web_url=data.get("URL"),
                keywords=data.get("subject", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse CrossRef paper: {e}")
            return None


class SearchService:
    """Unified search service for academic literature."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engines = {
            SearchEngine.SEMANTIC_SCHOLAR: SemanticScholarEngine(settings),
            SearchEngine.GOOGLE_SCHOLAR: GoogleScholarEngine(settings),
            # SearchEngine.WEB_OF_SCIENCE: WebOfScienceEngine(settings),  # Could be added
        }
        
        # Add CrossRef as additional engine
        self.crossref_engine = CrossRefEngine(settings)
    
    async def search(
        self,
        query: str,
        max_results: int = 50,
        engines: Optional[List[SearchEngine]] = None,
        **kwargs: Any
    ) -> List[LiteraturePaper]:
        """Search across multiple engines."""
        engines = engines or [SearchEngine.SEMANTIC_SCHOLAR, SearchEngine.GOOGLE_SCHOLAR]
        
        all_papers = []
        
        # Search each engine
        for engine in engines:
            if engine in self.engines:
                try:
                    engine_papers = await self.engines[engine].search(
                        query, 
                        max_results=max_results // len(engines),
                        **kwargs
                    )
                    all_papers.extend(engine_papers)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Search failed for {engine.value}: {e}")
        
        # Deduplicate papers
        unique_papers = self._deduplicate_papers(all_papers)
        
        logger.info(
            f"Search completed: {len(all_papers)} total papers, {len(unique_papers)} unique papers",
            query=query,
            engines=[e.value for e in engines]
        )
        
        return unique_papers
    
    def _deduplicate_papers(self, papers: List[LiteraturePaper]) -> List[LiteraturePaper]:
        """Remove duplicate papers."""
        seen_titles = set()
        seen_dois = set()
        unique_papers = []
        
        for paper in papers:
            # Check DOI first (most reliable)
            if paper.doi and paper.doi in seen_dois:
                continue
            
            # Check title similarity
            normalized_title = paper.title.lower().strip()
            if normalized_title in seen_titles:
                continue
            
            # Check for similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
                if paper.doi:
                    seen_dois.add(paper.doi)
        
        return unique_papers
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.85) -> bool:
        """Check if two titles are similar."""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union >= threshold
    
    async def search_by_doi(self, doi: str) -> Optional[LiteraturePaper]:
        """Search for a specific paper by DOI."""
        try:
            papers = await self.crossref_engine.search(f"doi:{doi}", max_results=1)
            return papers[0] if papers else None
        except Exception as e:
            logger.error(f"DOI search failed for {doi}: {e}")
            return None
    
    async def search_by_title(self, title: str) -> List[LiteraturePaper]:
        """Search for papers by exact title."""
        return await self.search(f'title:"{title}"', max_results=10)
    
    async def close(self) -> None:
        """Close all engine connections."""
        for engine in self.engines.values():
            await engine.close()
        await self.crossref_engine.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 
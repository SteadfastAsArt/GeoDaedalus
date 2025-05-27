"""Document processing service for GeoDaedalus."""

import asyncio
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from geodaedalus.core.config import Settings
from geodaedalus.core.logging import get_logger
from geodaedalus.core.models import LiteraturePaper

logger = get_logger(__name__)


class DocumentProcessor:
    """Base document processor."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "GeoDaedalus/1.0 (Academic Research Bot)",
            }
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()


class PDFProcessor(DocumentProcessor):
    """PDF document processor."""
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def extract_text(self, pdf_source: Union[str, Path, bytes]) -> str:
        """Extract text from PDF."""
        try:
            if isinstance(pdf_source, str) and pdf_source.startswith(('http://', 'https://')):
                # Download PDF from URL
                pdf_content = await self._download_pdf(pdf_source)
                return await self._extract_text_from_bytes(pdf_content)
            elif isinstance(pdf_source, Path):
                # Read from local file
                async with aiofiles.open(pdf_source, 'rb') as f:
                    pdf_content = await f.read()
                return await self._extract_text_from_bytes(pdf_content)
            elif isinstance(pdf_source, bytes):
                # Direct bytes
                return await self._extract_text_from_bytes(pdf_source)
            else:
                raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")
        
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        response = await self.http_client.get(url)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.warning(f"URL may not be a PDF: {url} (content-type: {content_type})")
        
        # Check file size
        content_length = int(response.headers.get('content-length', 0))
        max_size = self.settings.processing.max_file_size_mb * 1024 * 1024
        
        if content_length > max_size:
            raise ValueError(f"PDF too large: {content_length} bytes > {max_size} bytes")
        
        return response.content
    
    async def _extract_text_from_bytes(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes."""
        text = ""
        
        # Try PyPDF first (faster)
        try:
            reader = PdfReader(io.BytesIO(pdf_content))
            
            # Limit number of pages
            max_pages = min(len(reader.pages), self.settings.processing.max_pdf_pages)
            
            for page_num in range(max_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            if text.strip():
                logger.info(f"PyPDF extracted {len(text)} characters from {max_pages} pages")
                return text
        
        except Exception as e:
            logger.warning(f"PyPDF extraction failed: {e}, trying pdfminer")
        
        # Fallback to pdfminer (more robust)
        try:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                all_texts=False
            )
            
            text = pdfminer_extract_text(
                io.BytesIO(pdf_content),
                laparams=laparams,
                maxpages=self.settings.processing.max_pdf_pages
            )
            
            logger.info(f"pdfminer extracted {len(text)} characters")
            return text
        
        except Exception as e:
            logger.error(f"pdfminer extraction failed: {e}")
            return ""
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from PDF text."""
        sections = {}
        
        # Common section headers
        section_patterns = [
            r'\b(Abstract|ABSTRACT)\b',
            r'\b(Introduction|INTRODUCTION)\b',
            r'\b(Methods?|METHODS?|Methodology|METHODOLOGY)\b',
            r'\b(Results?|RESULTS?)\b',
            r'\b(Discussion|DISCUSSION)\b',
            r'\b(Conclusions?|CONCLUSIONS?)\b',
            r'\b(References?|REFERENCES?)\b',
            r'\b(Acknowledgments?|ACKNOWLEDGMENTS?)\b',
        ]
        
        # Find section boundaries
        section_matches = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                section_matches.append((match.start(), match.group().lower(), match.group()))
        
        # Sort by position
        section_matches.sort(key=lambda x: x[0])
        
        # Extract sections
        for i, (start, section_key, section_header) in enumerate(section_matches):
            # Find end of section (start of next section or end of text)
            if i + 1 < len(section_matches):
                end = section_matches[i + 1][0]
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            if section_text:
                sections[section_key] = section_text
        
        return sections
    
    def extract_tables_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract table information from PDF text."""
        tables = []
        
        # Look for table patterns
        table_patterns = [
            r'Table\s+(\d+)[.:]\s*([^\n]+)',
            r'TABLE\s+(\d+)[.:]\s*([^\n]+)',
        ]
        
        for pattern in table_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                table_num = match.group(1)
                caption = match.group(2).strip()
                
                tables.append({
                    "number": table_num,
                    "caption": caption,
                    "position": match.start(),
                    "type": "detected"
                })
        
        return tables


class HTMLProcessor(DocumentProcessor):
    """HTML document processor."""
    
    async def extract_content(self, html_source: Union[str, bytes]) -> Tuple[str, BeautifulSoup]:
        """Extract content from HTML."""
        try:
            if isinstance(html_source, str) and html_source.startswith(('http://', 'https://')):
                # Download HTML from URL
                html_content = await self._download_html(html_source)
                soup = BeautifulSoup(html_content, 'lxml')
            elif isinstance(html_source, bytes):
                soup = BeautifulSoup(html_source, 'lxml')
                html_content = html_source.decode('utf-8', errors='ignore')
            else:
                # Assume it's HTML string
                soup = BeautifulSoup(html_source, 'lxml')
                html_content = html_source
            
            # Extract main text content
            text_content = self._extract_main_text(soup)
            
            return text_content, soup
        
        except Exception as e:
            logger.error(f"HTML content extraction failed: {e}")
            return "", BeautifulSoup("", 'lxml')
    
    async def _download_html(self, url: str) -> bytes:
        """Download HTML from URL."""
        response = await self.http_client.get(url)
        response.raise_for_status()
        return response.content
    
    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML."""
        # Remove script, style, and other non-content tags
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "menu"]):
            tag.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Look for common content containers
        content_selectors = [
            "main", "article", ".content", "#content", ".main", "#main",
            ".article", ".paper", ".publication", ".abstract", ".full-text"
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements and elements[0].get_text(strip=True):
                main_content = elements[0]
                break
        
        # Fallback to body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        return clean_text
    
    def extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from HTML."""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            # Extract table data
            rows = []
            headers = []
            
            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            tbody = table.find('tbody')
            if tbody:
                # If we have a tbody, use all its rows
                for row in tbody.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
            else:
                # If no tbody, use all tr except the first one if it contains headers
                all_rows = table.find_all('tr')
                start_index = 1 if headers else 0
                for row in all_rows[start_index:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
            
            if headers or rows:
                # Find caption
                caption = ""
                caption_elem = table.find_previous('caption') or table.find('caption')
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)
                
                tables.append({
                    "index": i,
                    "caption": caption,
                    "headers": headers,
                    "data": rows,
                    "row_count": len(rows),
                    "col_count": len(headers) if headers else (len(rows[0]) if rows else 0)
                })
        
        return tables
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                if name in ['description', 'abstract']:
                    metadata['description'] = content
                elif name in ['author', 'authors']:
                    metadata['authors'] = content
                elif name in ['keywords']:
                    metadata['keywords'] = content.split(',')
                elif name == 'doi':
                    metadata['doi'] = content
        
        # Extract structured data (JSON-LD)
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                import json
                structured_data = json.loads(json_ld.string)
                if isinstance(structured_data, dict):
                    metadata['structured_data'] = structured_data
            except Exception:
                pass
        
        return metadata


class DocumentProcessorService:
    """Unified document processing service."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pdf_processor = PDFProcessor(settings)
        self.html_processor = HTMLProcessor(settings)
    
    async def process_paper(self, paper: LiteraturePaper) -> Dict[str, Any]:
        """Process a paper document."""
        result = {
            "paper_id": str(paper.id),
            "text_content": "",
            "sections": {},
            "tables": [],
            "metadata": {},
            "source_type": "unknown",
            "processing_status": "failed"
        }
        
        try:
            # Try PDF first if available
            if paper.pdf_url:
                result["source_type"] = "pdf"
                text_content = await self.pdf_processor.extract_text(str(paper.pdf_url))
                
                if text_content:
                    result["text_content"] = text_content
                    result["sections"] = self.pdf_processor.extract_sections(text_content)
                    result["tables"] = self.pdf_processor.extract_tables_info(text_content)
                    result["processing_status"] = "success"
                    return result
            
            # Fallback to web page
            if paper.web_url:
                result["source_type"] = "html"
                text_content, soup = await self.html_processor.extract_content(str(paper.web_url))
                
                if text_content:
                    result["text_content"] = text_content
                    result["tables"] = self.html_processor.extract_tables(soup)
                    result["metadata"] = self.html_processor.extract_metadata(soup)
                    result["processing_status"] = "success"
                    return result
            
            # Use abstract as fallback
            if paper.abstract:
                result["source_type"] = "abstract"
                result["text_content"] = paper.abstract
                result["sections"] = {"abstract": paper.abstract}
                result["processing_status"] = "partial"
                return result
        
        except Exception as e:
            logger.error(f"Document processing failed for paper {paper.id}: {e}")
            result["error"] = str(e)
        
        return result
    
    async def process_papers_batch(
        self, 
        papers: List[LiteraturePaper], 
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Process multiple papers concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(paper: LiteraturePaper) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_paper(paper)
        
        tasks = [process_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Paper processing failed for {papers[i].id}: {result}")
                processed_results.append({
                    "paper_id": str(papers[i].id),
                    "text_content": "",
                    "sections": {},
                    "tables": [],
                    "metadata": {},
                    "source_type": "unknown",
                    "processing_status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def extract_geochemical_data(self, text: str) -> Dict[str, Any]:
        """Extract geochemical data patterns from text."""
        data = {
            "elements_mentioned": [],
            "element_values": {},
            "rock_types": [],
            "locations": [],
            "analytical_methods": []
        }
        
        # Element patterns
        element_patterns = {
            "major_oxides": r'\b(SiO2|Al2O3|Fe2O3|FeO|MgO|CaO|Na2O|K2O|TiO2|P2O5|MnO)\b',
            "trace_elements": r'\b(Rb|Sr|Ba|Y|Zr|Nb|Th|U|V|Cr|Ni|Co|Sc|La|Ce|Nd|Sm|Eu|Gd|Dy|Er|Yb|Lu)\b',
            "isotope_ratios": r'\b(\d{2,3}[A-Z][a-z]?/\d{2,3}[A-Z][a-z]?)\b'
        }
        
        for category, pattern in element_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            data["elements_mentioned"].extend([(elem, category) for elem in matches])
        
        # Extract numerical values associated with elements
        value_pattern = r'(\w+)\s*[=:]\s*([\d.,]+)\s*([%]|ppm|ppb|wt%|mol%)?'
        for match in re.finditer(value_pattern, text):
            element, value, unit = match.groups()
            try:
                numeric_value = float(value.replace(',', ''))
                data["element_values"][element] = {
                    "value": numeric_value,
                    "unit": unit or "unknown"
                }
            except ValueError:
                pass
        
        # Rock type patterns (including plurals)
        rock_patterns = r'\b(basalts?|granites?|andesites?|rhyolites?|gabbros?|diorites?|sandstones?|limestones?|shales?|schists?|gneisses?)\b'
        data["rock_types"] = list(set(re.findall(rock_patterns, text, re.IGNORECASE)))
        
        # Analytical methods
        method_patterns = r'\b(XRF|ICP-MS|SIMS|EMP|SEM|TEM|Raman|FTIR|mass spectrometry)\b'
        data["analytical_methods"] = list(set(re.findall(method_patterns, text, re.IGNORECASE)))
        
        return data
    
    async def close(self) -> None:
        """Close all processors."""
        await self.pdf_processor.close()
        await self.html_processor.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 
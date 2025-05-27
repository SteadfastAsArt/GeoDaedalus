"""Agent 3: Data Extraction - Extracts data from PDFs and web pages with table detection."""

import asyncio
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from geodaedalus.agents.base import BaseAgent
from geodaedalus.core.models import (
    LiteraturePaper,
    DataTable,
    SearchResults,
    ExtractedData,
    GeospatialLocation,
    RockType,
)
from geodaedalus.services.llm import LLMService


class DataExtractionAgent(BaseAgent[SearchResults, ExtractedData]):
    """Agent for extracting geochemical data from literature papers."""
    
    def __init__(self, **kwargs):
        """Initialize data extraction agent."""
        super().__init__("data_extraction", **kwargs)
        self.llm_service = LLMService(self.settings)
        self.http_client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        
        # Table detection patterns
        self.table_indicators = [
            r"table\s+\d+",
            r"tab\.\s*\d+",
            r"supplementary\s+table",
            r"data\s+table",
            r"geochemical\s+data",
            r"chemical\s+composition",
            r"element\s+concentrations?",
            r"oxide\s+compositions?",
        ]
        
        # Element patterns for geochemical data
        self.element_patterns = [
            r"\b(SiO2|TiO2|Al2O3|Fe2O3|FeO|MnO|MgO|CaO|Na2O|K2O|P2O5)\b",
            r"\b(Si|Ti|Al|Fe|Mn|Mg|Ca|Na|K|P)\b",
            r"\b(Rb|Sr|Y|Zr|Nb|Ba|La|Ce|Pr|Nd|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu)\b",
            r"\b(Sc|V|Cr|Co|Ni|Cu|Zn|Ga|As|Se|Br|Mo|Ag|Cd|In|Sn|Sb|Te|I|Cs|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Th|U)\b",
        ]
    
    async def process(
        self, 
        search_results: SearchResults, 
        max_papers: Optional[int] = None,
        **kwargs: Any
    ) -> ExtractedData:
        """Extract data from literature search results."""
        if not self.validate_input(search_results):
            raise ValueError("Invalid search results provided")
        
        papers_to_process = search_results.papers
        if max_papers:
            papers_to_process = papers_to_process[:max_papers]
        
        self.logger.info(
            "Starting data extraction",
            total_papers=len(papers_to_process),
            query_id=str(search_results.query_id)
        )
        
        all_tables = []
        processed_papers = []
        
        # Process papers concurrently with rate limiting
        semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
        
        tasks = [
            self._process_paper_with_semaphore(semaphore, paper)
            for paper in papers_to_process
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Failed to process paper {papers_to_process[i].title}",
                    error=str(result)
                )
            elif result:
                tables, paper_id = result
                all_tables.extend(tables)
                processed_papers.append(paper_id)
        
        # Consolidate extracted data
        consolidated_data = await self._consolidate_data(all_tables)
        
        self.logger.info(
            "Data extraction completed",
            papers_processed=len(processed_papers),
            tables_extracted=len(all_tables),
            consolidated_samples=len(consolidated_data.get("samples", []))
        )
        
        return ExtractedData(
            query_id=search_results.query_id,
            tables=all_tables,
            consolidated_data=consolidated_data,
            source_papers=processed_papers,
            extraction_summary={
                "papers_processed": len(processed_papers),
                "tables_extracted": len(all_tables),
                "extraction_methods": ["pdf_text", "html_parsing", "llm_structured"]
            }
        )
    
    async def _process_paper_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        paper: LiteraturePaper
    ) -> Optional[Tuple[List[DataTable], str]]:
        """Process a single paper with semaphore for rate limiting."""
        async with semaphore:
            return await self._process_paper(paper)
    
    async def _process_paper(self, paper: LiteraturePaper) -> Optional[Tuple[List[DataTable], str]]:
        """Process a single paper to extract data tables."""
        try:
            self.logger.info(f"Processing paper: {paper.title[:100]}...")
            
            # Try PDF extraction first
            if paper.pdf_url:
                tables = await self._extract_from_pdf(paper)
                if tables:
                    return tables, str(paper.id)
            
            # Fallback to web page extraction
            if paper.web_url:
                tables = await self._extract_from_webpage(paper)
                if tables:
                    return tables, str(paper.id)
            
            # If no direct content, try LLM-based extraction from abstract
            if paper.abstract:
                tables = await self._extract_from_abstract(paper)
                if tables:
                    return tables, str(paper.id)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing paper {paper.title}", error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _extract_from_pdf(self, paper: LiteraturePaper) -> List[DataTable]:
        """Extract data tables from PDF."""
        try:
            # Download PDF
            response = await self.http_client.get(str(paper.pdf_url))
            response.raise_for_status()
            
            pdf_content = BytesIO(response.content)
            
            # Extract text using pdfminer
            text = extract_text(pdf_content, laparams=LAParams())
            
            # Also try pypdf for better table extraction
            pdf_reader = PdfReader(pdf_content)
            pages_text = []
            for page in pdf_reader.pages:
                pages_text.append(page.extract_text())
            
            # Combine extraction methods
            full_text = text + "\n\n" + "\n\n".join(pages_text)
            
            # Find potential table sections
            table_sections = self._identify_table_sections(full_text)
            
            # Extract tables using LLM
            tables = []
            for section in table_sections:
                extracted_tables = await self._extract_tables_with_llm(section, paper)
                tables.extend(extracted_tables)
            
            return tables
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed for {paper.title}", error=str(e))
            return []
    
    async def _extract_from_webpage(self, paper: LiteraturePaper) -> List[DataTable]:
        """Extract data tables from web page."""
        try:
            response = await self.http_client.get(str(paper.web_url))
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Look for HTML tables
            html_tables = soup.find_all('table')
            tables = []
            
            for i, html_table in enumerate(html_tables):
                if self._is_geochemical_table(html_table):
                    table = await self._parse_html_table(html_table, paper, i)
                    if table:
                        tables.append(table)
            
            # If no HTML tables found, extract text and look for tabular data
            if not tables:
                text = soup.get_text()
                table_sections = self._identify_table_sections(text)
                
                for section in table_sections:
                    extracted_tables = await self._extract_tables_with_llm(section, paper)
                    tables.extend(extracted_tables)
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Webpage extraction failed for {paper.title}", error=str(e))
            return []
    
    async def _extract_from_abstract(self, paper: LiteraturePaper) -> List[DataTable]:
        """Extract potential data from abstract using LLM."""
        try:
            # Only extract if abstract mentions specific geochemical data
            if not any(pattern in paper.abstract.lower() for pattern in [
                "geochemical", "chemical composition", "element", "oxide", "concentration"
            ]):
                return []
            
            prompt = f"""
            Extract any geochemical data mentioned in this abstract. Look for:
            - Element concentrations
            - Oxide compositions
            - Isotope ratios
            - Sample locations
            - Rock types
            
            Abstract: {paper.abstract}
            
            If you find specific numerical data, format it as a JSON table with:
            - headers: list of column names
            - data: list of rows (each row is a list of values)
            - sample_info: any sample identification or location info
            
            Return JSON or "NO_DATA" if no specific numerical data is found.
            """
            
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=800,
                temperature=0.1
            )
            
            if "NO_DATA" in response.content:
                return []
            
            try:
                data = json.loads(response.content)
                if data.get("headers") and data.get("data"):
                    table = DataTable(
                        paper_id=paper.id,
                        table_number="abstract",
                        caption=f"Data extracted from abstract of: {paper.title}",
                        headers=data["headers"],
                        data=data["data"],
                        sample_ids=data.get("sample_info", []),
                        extraction_confidence=0.6  # Lower confidence for abstract extraction
                    )
                    return [table]
            except json.JSONDecodeError:
                pass
            
            return []
            
        except Exception as e:
            self.logger.error(f"Abstract extraction failed for {paper.title}", error=str(e))
            return []
    
    def _identify_table_sections(self, text: str) -> List[str]:
        """Identify sections of text that likely contain tables."""
        sections = []
        lines = text.split('\n')
        
        current_section = []
        in_table_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line indicates start of table
            is_table_indicator = any(
                re.search(pattern, line.lower()) for pattern in self.table_indicators
            )
            
            # Check if line contains geochemical elements
            has_elements = any(
                re.search(pattern, line) for pattern in self.element_patterns
            )
            
            # Check if line looks like tabular data (multiple numbers)
            numbers = re.findall(r'\d+\.?\d*', line)
            is_tabular = len(numbers) >= 3
            
            if is_table_indicator or (has_elements and is_tabular):
                if current_section and not in_table_section:
                    # Start new table section
                    current_section = [line]
                    in_table_section = True
                else:
                    current_section.append(line)
                    in_table_section = True
            elif in_table_section:
                if is_tabular or has_elements:
                    current_section.append(line)
                else:
                    # End of table section
                    if len(current_section) >= 3:  # Minimum viable table
                        sections.append('\n'.join(current_section))
                    current_section = []
                    in_table_section = False
        
        # Add final section if it exists
        if current_section and len(current_section) >= 3:
            sections.append('\n'.join(current_section))
        
        return sections
    
    async def _extract_tables_with_llm(
        self, 
        text_section: str, 
        paper: LiteraturePaper
    ) -> List[DataTable]:
        """Extract structured tables from text using LLM."""
        try:
            prompt = f"""
            Extract geochemical data tables from this text section. Look for:
            - Sample identifiers
            - Element concentrations (major, minor, trace elements)
            - Oxide compositions (SiO2, Al2O3, etc.)
            - Location information
            - Rock type information
            
            Text section:
            {text_section}
            
            Format the output as JSON with this structure:
            {{
                "tables": [
                    {{
                        "caption": "description of the table",
                        "headers": ["Sample", "SiO2", "Al2O3", ...],
                        "data": [
                            ["Sample1", 65.2, 15.8, ...],
                            ["Sample2", 62.1, 16.2, ...]
                        ],
                        "sample_info": ["Sample1", "Sample2", ...],
                        "location": "location if mentioned",
                        "rock_type": "rock type if mentioned"
                    }}
                ]
            }}
            
            Return "NO_TABLES" if no structured data tables are found.
            """
            
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            if "NO_TABLES" in response.content:
                return []
            
            try:
                result = json.loads(response.content)
                tables = []
                
                for i, table_data in enumerate(result.get("tables", [])):
                    # Parse location if provided
                    location = None
                    if table_data.get("location"):
                        location = GeospatialLocation(location_name=table_data["location"])
                    
                    # Parse rock type if provided
                    rock_type = None
                    if table_data.get("rock_type"):
                        try:
                            rock_type = RockType(table_data["rock_type"].lower())
                        except ValueError:
                            pass
                    
                    # Extract element data
                    element_data = self._extract_element_data(
                        table_data.get("headers", []),
                        table_data.get("data", [])
                    )
                    
                    table = DataTable(
                        paper_id=paper.id,
                        table_number=f"extracted_{i+1}",
                        caption=table_data.get("caption", "Extracted geochemical data"),
                        headers=table_data.get("headers", []),
                        data=table_data.get("data", []),
                        sample_ids=table_data.get("sample_info", []),
                        location_info=location,
                        rock_type=rock_type,
                        element_data=element_data,
                        extraction_confidence=0.8
                    )
                    tables.append(table)
                
                return tables
                
            except json.JSONDecodeError as e:
                self.logger.error("Failed to parse LLM table extraction response", error=str(e))
                return []
            
        except Exception as e:
            self.logger.error("LLM table extraction failed", error=str(e))
            return []
    
    def _is_geochemical_table(self, html_table) -> bool:
        """Check if HTML table contains geochemical data."""
        table_text = html_table.get_text().lower()
        
        # Check for geochemical indicators
        indicators = [
            "sio2", "al2o3", "fe2o3", "mgo", "cao", "na2o", "k2o",
            "element", "oxide", "concentration", "ppm", "wt%", "weight%",
            "geochemical", "chemical composition", "major element", "trace element"
        ]
        
        return any(indicator in table_text for indicator in indicators)
    
    async def _parse_html_table(
        self, 
        html_table, 
        paper: LiteraturePaper, 
        table_index: int
    ) -> Optional[DataTable]:
        """Parse HTML table into DataTable format."""
        try:
            # Extract table data using pandas
            df = pd.read_html(str(html_table))[0]
            
            # Clean up the dataframe
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                return None
            
            # Convert to our format
            headers = [str(col) for col in df.columns]
            data = df.values.tolist()
            
            # Extract sample IDs (usually first column)
            sample_ids = []
            if headers and data:
                sample_ids = [str(row[0]) for row in data if row]
            
            # Extract element data
            element_data = self._extract_element_data(headers, data)
            
            return DataTable(
                paper_id=paper.id,
                table_number=f"html_{table_index+1}",
                caption=f"Table {table_index+1} from {paper.title}",
                headers=headers,
                data=data,
                sample_ids=sample_ids,
                element_data=element_data,
                extraction_confidence=0.9
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse HTML table {table_index}", error=str(e))
            return None
    
    def _extract_element_data(self, headers: List[str], data: List[List[Any]]) -> Dict[str, List[float]]:
        """Extract element concentration data from table."""
        element_data = {}
        
        if not headers or not data:
            return element_data
        
        # Find columns that contain element data
        for i, header in enumerate(headers):
            header_clean = str(header).strip()
            
            # Check if header matches element patterns
            is_element = any(
                re.search(pattern, header_clean, re.IGNORECASE) 
                for pattern in self.element_patterns
            )
            
            if is_element:
                # Extract numeric values from this column
                values = []
                for row in data:
                    if i < len(row):
                        try:
                            # Try to convert to float
                            value = str(row[i]).strip()
                            # Remove common non-numeric characters
                            value = re.sub(r'[<>±\s]', '', value)
                            if value and value != 'nan':
                                values.append(float(value))
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    element_data[header_clean] = values
        
        return element_data
    
    async def _consolidate_data(self, tables: List[DataTable]) -> Dict[str, Any]:
        """Consolidate data from multiple tables."""
        if not tables:
            return {}
        
        # Group tables by paper
        papers_data = {}
        for table in tables:
            paper_id = str(table.paper_id)
            if paper_id not in papers_data:
                papers_data[paper_id] = []
            papers_data[paper_id].append(table)
        
        # Consolidate element data
        all_elements = set()
        all_samples = []
        
        for table in tables:
            all_elements.update(table.element_data.keys())
            all_samples.extend(table.sample_ids)
        
        # Create consolidated dataset
        consolidated = {
            "total_tables": len(tables),
            "total_papers": len(papers_data),
            "total_samples": len(set(all_samples)),
            "elements_found": list(all_elements),
            "samples": all_samples,
            "papers_data": papers_data,
            "element_statistics": self._calculate_element_statistics(tables)
        }
        
        return consolidated
    
    def _calculate_element_statistics(self, tables: List[DataTable]) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for element concentrations."""
        element_stats = {}
        
        # Collect all values for each element
        element_values = {}
        for table in tables:
            for element, values in table.element_data.items():
                if element not in element_values:
                    element_values[element] = []
                element_values[element].extend(values)
        
        # Calculate statistics
        for element, values in element_values.items():
            if values:
                element_stats[element] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2]
                }
        
        return element_stats
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose() 
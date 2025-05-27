"""Tests for GeoDaedalus services."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from geodaedalus.core.config import Settings, ProcessingConfig
from geodaedalus.core.models import Author, LiteraturePaper, SearchEngine
from geodaedalus.services.search import SearchService, SemanticScholarEngine, GoogleScholarEngine
from geodaedalus.services.document_processor import DocumentProcessorService, PDFProcessor, HTMLProcessor


class TestSearchService:
    """Test cases for SearchService."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            openai_api_key="test-key",
            serpapi_key="test-serpapi-key"
        )
    
    @pytest.fixture
    def search_service(self, settings):
        """Create SearchService instance."""
        return SearchService(settings)
    
    @pytest.mark.asyncio
    async def test_semantic_scholar_search(self, settings):
        """Test Semantic Scholar search engine."""
        engine = SemanticScholarEngine(settings)
        
        mock_response_data = {
            "data": [
                {
                    "paperId": "test-paper-1",
                    "title": "Test Paper Title",
                    "authors": [{"name": "John Doe"}],
                    "abstract": "Test abstract content",
                    "year": 2023,
                    "venue": "Test Journal",
                    "citationCount": 42,
                    "isOpenAccess": True,
                    "url": "https://test.url",
                    "externalIds": {"DOI": "10.1000/test"}
                }
            ]
        }
        
        with patch.object(engine.http_client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status = AsyncMock()
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_get.return_value = mock_response
            
            papers = await engine.search("geochemistry", max_results=10)
            
            assert len(papers) == 1
            paper = papers[0]
            assert paper.title == "Test Paper Title"
            assert len(paper.authors) == 1
            assert paper.authors[0].name == "John Doe"
            assert paper.doi == "10.1000/test"
            assert paper.citation_count == 42
        
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_google_scholar_search(self, settings):
        """Test Google Scholar search engine."""
        engine = GoogleScholarEngine(settings)
        
        mock_response_data = {
            "organic_results": [
                {
                    "title": "Google Scholar Paper",
                    "snippet": "Paper abstract from Google Scholar",
                    "link": "https://scholar.google.com/test",
                    "publication_info": {
                        "authors": "Jane Smith, Bob Wilson",
                        "summary": "Nature Geoscience, 2022"
                    },
                    "inline_links": {
                        "cited_by": {"total": 25}
                    }
                }
            ]
        }
        
        with patch.object(engine.http_client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status = AsyncMock()
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_get.return_value = mock_response
            
            papers = await engine.search("geochemistry", max_results=10)
            
            assert len(papers) == 1
            paper = papers[0]
            assert paper.title == "Google Scholar Paper"
            assert len(paper.authors) == 2
            assert paper.authors[0].name == "Jane Smith"
            assert paper.year == 2022
        
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_search_service_integration(self, search_service):
        """Test SearchService with multiple engines."""
        # Mock both engines
        mock_semantic_papers = [
            LiteraturePaper(
                title="Semantic Scholar Paper",
                authors=[Author(name="Author 1")],
                abstract="Abstract 1"
            )
        ]
        
        mock_google_papers = [
            LiteraturePaper(
                title="Google Scholar Paper",
                authors=[Author(name="Author 2")],
                abstract="Abstract 2"
            )
        ]
        
        with patch.object(search_service.engines[SearchEngine.SEMANTIC_SCHOLAR], 'search', return_value=mock_semantic_papers), \
             patch.object(search_service.engines[SearchEngine.GOOGLE_SCHOLAR], 'search', return_value=mock_google_papers):
            
            papers = await search_service.search(
                "geochemistry",
                max_results=20,
                engines=[SearchEngine.SEMANTIC_SCHOLAR, SearchEngine.GOOGLE_SCHOLAR]
            )
            
            assert len(papers) == 2
            titles = [p.title for p in papers]
            assert "Semantic Scholar Paper" in titles
            assert "Google Scholar Paper" in titles
        
        await search_service.close()
    
    @pytest.mark.asyncio
    async def test_duplicate_removal(self, search_service):
        """Test duplicate paper removal."""
        # Create papers with similar titles
        papers = [
            LiteraturePaper(
                title="Geochemical Analysis of Volcanic Rocks",
                authors=[Author(name="Author 1")],
                abstract="Abstract 1"
            ),
            LiteraturePaper(
                title="geochemical analysis of volcanic rocks",  # Same title, different case
                authors=[Author(name="Author 2")],
                abstract="Abstract 2"
            ),
            LiteraturePaper(
                title="Petrology of Igneous Rocks",
                authors=[Author(name="Author 3")],
                abstract="Abstract 3"
            )
        ]
        
        unique_papers = search_service._deduplicate_papers(papers)
        
        assert len(unique_papers) == 2  # Should remove one duplicate
        titles = [p.title.lower() for p in unique_papers]
        assert "geochemical analysis of volcanic rocks" in titles
        assert "petrology of igneous rocks" in titles
    
    def test_title_similarity(self, search_service):
        """Test title similarity detection."""
        title1 = "geochemical analysis of volcanic rocks"
        title2 = "geochemical analysis of volcanic rocks and minerals"
        title3 = "petrology of igneous formations"
        
        # Should be similar
        assert search_service._titles_similar(title1, title2, threshold=0.7)
        
        # Should not be similar
        assert not search_service._titles_similar(title1, title3, threshold=0.7)


class TestDocumentProcessorService:
    """Test cases for DocumentProcessorService."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        from geodaedalus.core.config import ProcessingConfig
        return Settings(
            processing=ProcessingConfig(
                max_file_size_mb=10,
                max_pdf_pages=50
            )
        )
    
    @pytest.fixture
    def doc_processor(self, settings):
        """Create DocumentProcessorService instance."""
        return DocumentProcessorService(settings)
    
    @pytest.mark.asyncio
    async def test_pdf_text_extraction(self, settings):
        """Test PDF text extraction."""
        processor = PDFProcessor(settings)
        
        # Mock PDF content
        mock_pdf_content = b"%PDF-1.4 mock content"
        
        with patch('geodaedalus.services.document_processor.PdfReader') as mock_reader_class:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Extracted text from PDF"
            mock_reader.pages = [mock_page]
            mock_reader_class.return_value = mock_reader
            
            text = await processor._extract_text_from_bytes(mock_pdf_content)
            
            assert "Extracted text from PDF" in text
        
        await processor.close()
    
    def test_pdf_section_extraction(self, settings):
        """Test PDF section extraction."""
        processor = PDFProcessor(settings)
        
        text = """
        Abstract
        This paper presents a geochemical analysis of volcanic rocks.
        
        Introduction
        Volcanic rocks are important for understanding magmatic processes.
        
        Methods
        Samples were analyzed using XRF and ICP-MS.
        
        Results
        The data show significant variations in element concentrations.
        """
        
        sections = processor.extract_sections(text)
        
        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "This paper presents" in sections["abstract"]
    
    def test_html_table_extraction(self, settings):
        """Test HTML table extraction."""
        processor = HTMLProcessor(settings)
        
        html_content = """
        <html>
        <body>
            <table>
                <caption>Table 1. Geochemical data</caption>
                <thead>
                    <tr>
                        <th>Sample</th>
                        <th>SiO2</th>
                        <th>Al2O3</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>S1</td>
                        <td>49.2</td>
                        <td>13.8</td>
                    </tr>
                    <tr>
                        <td>S2</td>
                        <td>47.8</td>
                        <td>14.2</td>
                    </tr>
                </tbody>
            </table>
        </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, 'lxml')
        tables = processor.extract_tables(soup)
        
        assert len(tables) == 1
        table = tables[0]
        assert table["caption"] == "Table 1. Geochemical data"
        assert table["headers"] == ["Sample", "SiO2", "Al2O3"]
        assert len(table["data"]) == 2
        assert table["data"][0] == ["S1", "49.2", "13.8"]
    
    def test_geochemical_data_extraction(self, doc_processor):
        """Test geochemical data pattern extraction."""
        text = """
        Major element concentrations were measured using XRF.
        SiO2 = 49.2%, Al2O3 = 13.8%, MgO = 8.4%.
        Trace elements were analyzed by ICP-MS.
        Rb = 15 ppm, Sr = 380 ppm, Ba = 250 ppm.
        The rocks are classified as basalts from Iceland.
        """
        
        data = doc_processor.extract_geochemical_data(text)
        
        # Check elements mentioned
        element_names = [elem[0] for elem in data["elements_mentioned"]]
        assert "SiO2" in element_names
        assert "Al2O3" in element_names
        assert "Rb" in element_names
        
        # Check element values
        assert "SiO2" in data["element_values"]
        assert data["element_values"]["SiO2"]["value"] == 49.2
        assert data["element_values"]["SiO2"]["unit"] == "%"
        
        # Check rock types
        assert "basalts" in [rt.lower() for rt in data["rock_types"]]
        
        # Check analytical methods
        assert "XRF" in data["analytical_methods"]
        assert "ICP-MS" in data["analytical_methods"]
    
    @pytest.mark.asyncio
    async def test_paper_processing_workflow(self, doc_processor):
        """Test complete paper processing workflow."""
        paper = LiteraturePaper(
            title="Test Paper",
            authors=[Author(name="Test Author")],
            abstract="This paper studies geochemical processes in volcanic rocks.",
            web_url="https://example.com/paper.html",
            pdf_url=None
        )
        
        # Mock HTML content
        html_content = """
        <html>
        <body>
            <h1>Test Paper</h1>
            <p>Abstract: This paper studies geochemical processes in volcanic rocks.</p>
            <table>
                <tr><th>Sample</th><th>SiO2</th></tr>
                <tr><td>S1</td><td>49.2</td></tr>
            </table>
        </body>
        </html>
        """
        
        with patch.object(doc_processor.html_processor, 'extract_content') as mock_extract:
            soup = BeautifulSoup(html_content, 'lxml')
            mock_extract.return_value = ("Test paper content with geochemical data", soup)
            
            result = await doc_processor.process_paper(paper)
            
            assert result["processing_status"] == "success"
            assert result["source_type"] == "html"
            assert len(result["text_content"]) > 0
            assert len(result["tables"]) > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, doc_processor):
        """Test batch paper processing."""
        papers = [
            LiteraturePaper(
                title="Paper 1",
                authors=[Author(name="Author 1")],
                abstract="Abstract 1"
            ),
            LiteraturePaper(
                title="Paper 2", 
                authors=[Author(name="Author 2")],
                abstract="Abstract 2"
            )
        ]
        
        with patch.object(doc_processor, 'process_paper') as mock_process:
            mock_process.side_effect = [
                {
                    "paper_id": "1",
                    "processing_status": "success",
                    "text_content": "Content 1",
                    "tables": [],
                    "sections": {"abstract": "Abstract 1"},
                    "metadata": {},
                    "source_type": "abstract"
                },
                {
                    "paper_id": "2",
                    "processing_status": "success", 
                    "text_content": "Content 2",
                    "tables": [],
                    "sections": {"abstract": "Abstract 2"},
                    "metadata": {},
                    "source_type": "abstract"
                }
            ]
            
            results = await doc_processor.process_papers_batch(papers, max_concurrent=2)
            
            assert len(results) == 2
            assert all(r["processing_status"] == "success" for r in results)


class TestBenchmarkDatasets:
    """Test cases for benchmark datasets."""
    
    def test_intent_understanding_dataset(self):
        """Test intent understanding dataset generation."""
        from geodaedalus.benchmark.datasets import IntentUnderstandingDataset, DifficultyLevel
        
        dataset = IntentUnderstandingDataset()
        samples = dataset.generate_samples(count=20)
        
        assert len(samples) == 20
        assert all(sample.task.value == "intent_understanding" for sample in samples)
        
        # Check difficulty distribution
        difficulties = [sample.difficulty for sample in samples]
        assert DifficultyLevel.EASY in difficulties
        assert DifficultyLevel.MEDIUM in difficulties
    
    def test_benchmark_sample_serialization(self):
        """Test benchmark sample serialization."""
        from geodaedalus.benchmark.datasets import BenchmarkSample, BenchmarkTask, DifficultyLevel
        
        sample = BenchmarkSample(
            id="test-1",
            task=BenchmarkTask.INTENT_UNDERSTANDING,
            difficulty=DifficultyLevel.EASY,
            input_data={"user_query": "Find volcanic rocks"},
            expected_output={"rock_types": ["volcanic"]},
            metadata={"test": True}
        )
        
        # Test serialization
        sample_dict = sample.to_dict()
        assert sample_dict["task"] == "intent_understanding"
        assert sample_dict["difficulty"] == "easy"
        
        # Test deserialization
        restored_sample = BenchmarkSample.from_dict(sample_dict)
        assert restored_sample.id == sample.id
        assert restored_sample.task == sample.task
        assert restored_sample.difficulty == sample.difficulty
    
    def test_dataset_file_operations(self, tmp_path):
        """Test dataset file save/load operations."""
        from geodaedalus.benchmark.datasets import DataExtractionDataset
        
        dataset = DataExtractionDataset()
        dataset.generate_samples(count=5)
        
        # Save to file
        file_path = tmp_path / "test_dataset.json"
        dataset.save_to_file(file_path)
        
        assert file_path.exists()
        
        # Load from file
        new_dataset = DataExtractionDataset()
        new_dataset.load_from_file(file_path)
        
        assert len(new_dataset.samples) == 5
        assert new_dataset.samples[0].task.value == "data_extraction"


# Integration tests
class TestServiceIntegration:
    """Integration tests for multiple services."""
    
    @pytest.mark.asyncio
    async def test_search_to_processing_pipeline(self):
        """Test integration between search and document processing."""
        settings = Settings(openai_api_key="test-key")
        
        # Mock search results
        mock_papers = [
            LiteraturePaper(
                title="Test Geochemistry Paper",
                authors=[Author(name="Test Author")],
                abstract="This paper analyzes volcanic rock geochemistry.",
                web_url="https://example.com/paper.html"
            )
        ]
        
        search_service = SearchService(settings)
        doc_processor = DocumentProcessorService(settings)
        
        # Mock search
        with patch.object(search_service, 'search', return_value=mock_papers):
            papers = await search_service.search("volcanic geochemistry")
            
            assert len(papers) == 1
            
            # Mock document processing
            with patch.object(doc_processor, 'process_paper') as mock_process:
                mock_process.return_value = {
                    "paper_id": str(papers[0].id),
                    "processing_status": "success",
                    "text_content": "Geochemical analysis results...",
                    "tables": [{"headers": ["Sample", "SiO2"], "data": [["S1", "49.2"]]}],
                    "sections": {"abstract": papers[0].abstract},
                    "metadata": {},
                    "source_type": "html"
                }
                
                result = await doc_processor.process_paper(papers[0])
                
                assert result["processing_status"] == "success"
                assert len(result["tables"]) > 0
        
        await search_service.close()
        await doc_processor.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 
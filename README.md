# GeoDaedalus 🌍🔬

A comprehensive AI-powered geoscientific data extraction pipeline that intelligently searches, processes, and fuses geochemical data from academic literature.

## ✨ Features

### 🧠 **Intelligent Natural Language Understanding**
- Parse complex geoscientific queries in natural language
- Extract spatial, temporal, and geochemical constraints
- Support for geological periods, rock types, and element categories
- LLM-powered with domain-specific geological knowledge

### 📚 **Multi-Source Literature Search**
- **Semantic Scholar API** - Access to millions of academic papers
- **Google Scholar** via SerpAPI - Comprehensive academic search
- **CrossRef API** - DOI-based paper retrieval
- Advanced deduplication and relevance scoring
- Citation count and open access filtering

### 📄 **Robust Document Processing**
- **PDF Processing** - PyPDF + pdfminer fallback for text extraction
- **HTML Processing** - Web content extraction with table detection
- **Section Detection** - Automatic identification of Abstract, Methods, Results, etc.
- **Table Extraction** - Structured data extraction from documents
- **Geochemical Pattern Recognition** - Element detection and value extraction

### 🔬 **Advanced Data Fusion**
- Multi-source data integration and validation
- Duplicate detection and removal
- Outlier identification and quality assessment
- Confidence scoring and data completeness metrics
- Statistical validation of geochemical data

### 📊 **Comprehensive Benchmarking**
- Task-specific evaluation datasets (Intent Understanding, Literature Retrieval, Data Extraction, Data Fusion)
- Multiple difficulty levels (Easy, Medium, Hard, Expert)
- Automated performance metrics (Accuracy, Precision, Recall, F1-Score)
- Rich evaluation reports with visualizations

## 🏗️ Architecture

```
GeoDaedalus/
├── 🔧 Core Components
│   ├── ⚙️ Configuration & Settings
│   ├── 📝 Logging & Monitoring
│   ├── 🔄 Pipeline Orchestrator
│   └── 📊 Data Models & Schemas
├── 🤖 Intelligent Agents
│   ├── 🧠 Requirement Understanding Agent
│   ├── 🔍 Literature Search Agent
│   ├── 📄 Data Extraction Agent
│   └── 🔬 Data Fusion Agent
├── 🛠️ Services
│   ├── 🤖 LLM Service (OpenAI)
│   ├── 🔍 Academic Search Service
│   └── 📄 Document Processing Service
└── 📏 Evaluation & Benchmarking
    ├── 📊 Benchmark Datasets
    ├── 🎯 Task-Specific Evaluators
    ├── 📈 Performance Metrics
    └── 📋 Comprehensive Reporting
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/geodaedalus.git
cd geodaedalus
pip install -e .
```

### Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
SERPAPI_KEY=your_serpapi_key
```

### Run the Demo

```bash
# Quick demo with a single query
python demo.py --mode quick

# Full demo with multiple queries
python demo.py --mode full

# Benchmark evaluation demo
python demo.py --mode benchmark
```

### CLI Usage

```bash
# Interactive query processing
python -m geodaedalus.cli.main query "Find volcanic rocks from Hawaii with major elements"

# Batch processing
python -m geodaedalus.cli.main batch --input queries.txt --output results.json

# Generate benchmark datasets
python -m geodaedalus.cli.main benchmark --generate --output ./benchmarks

# Run evaluation
python -m geodaedalus.cli.main benchmark --evaluate --dataset ./benchmarks --output ./results
```

## 📖 Usage Examples

### Python API

```python
import asyncio
from geodaedalus.core.pipeline import GeoDaedalusPipeline
from geodaedalus.core.config import Settings

async def main():
    settings = Settings()
    pipeline = GeoDaedalusPipeline(settings)
    
    # Run complete pipeline
    result = await pipeline.run_complete_pipeline(
        user_query="Find Cretaceous basalts from Deccan Traps with trace elements",
        max_papers=10
    )
    
    print(f"Found {len(result['papers'])} papers")
    print(f"Extracted {len(result['extracted_data'])} datasets")
    print(f"Quality score: {result['quality_metrics']['overall_quality']:.2f}")
    
    await pipeline.close()

asyncio.run(main())
```

### Step-by-Step Processing

```python
# 1. Parse natural language query
constraints = await pipeline.requirement_agent.process(
    "Find volcanic rocks from Hawaii with major element compositions"
)

# 2. Search literature
papers = await pipeline.literature_agent.process(constraints, max_papers=20)

# 3. Extract data from documents
extracted_data = await pipeline.extraction_agent.process(papers)

# 4. Fuse and validate data
fused_data = await pipeline.fusion_agent.process(extracted_data)
```

## 🔬 Supported Data Types

### Geochemical Elements
- **Major Elements**: SiO₂, Al₂O₃, Fe₂O₃, MgO, CaO, Na₂O, K₂O, TiO₂, P₂O₅, MnO
- **Trace Elements**: Rb, Sr, Ba, Y, Zr, Nb, Th, U, V, Cr, Ni, Co, Sc
- **Rare Earth Elements**: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
- **Isotope Ratios**: ⁸⁷Sr/⁸⁶Sr, ¹⁴³Nd/¹⁴⁴Nd, ²⁰⁶Pb/²⁰⁴Pb, etc.

### Rock Types
- **Igneous**: Volcanic, Plutonic, Basalt, Granite, Andesite, Rhyolite, Gabbro, Diorite
- **Sedimentary**: Clastic, Carbonate, Sandstone, Limestone, Shale
- **Metamorphic**: Schist, Gneiss, Amphibolite, Granulite

### Geological Context
- **Temporal**: Geological periods, absolute ages, stratigraphic units
- **Spatial**: Geographic locations, tectonic settings, geological formations
- **Environmental**: Pressure-temperature conditions, metamorphic grades

## 📊 Benchmarking

GeoDaedalus includes a comprehensive benchmarking system:

### Benchmark Tasks
1. **Intent Understanding** - Natural language to structured constraints
2. **Literature Retrieval** - Relevant paper finding and ranking
3. **Data Extraction** - Table and data extraction from documents
4. **Data Fusion** - Multi-source data integration and validation

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision/Recall**: Information retrieval quality
- **F1-Score**: Harmonic mean of precision and recall
- **Completeness**: Coverage of expected outputs
- **Execution Time**: Performance benchmarking
- **Cost Estimation**: API usage and resource consumption

### Generate and Run Benchmarks

```bash
# Generate benchmark datasets
python -c "
from geodaedalus.benchmark.datasets import GeoDataBenchDataset
from pathlib import Path

dataset = GeoDataBenchDataset()
dataset.generate_all_datasets(Path('benchmarks'))
print('Benchmark datasets generated!')
"

# Run comprehensive evaluation
python -c "
import asyncio
from geodaedalus.benchmark.evaluator import BenchmarkEvaluator
from geodaedalus.core.config import Settings
from pathlib import Path

async def evaluate():
    evaluator = BenchmarkEvaluator(Settings())
    report = await evaluator.run_comprehensive_evaluation(
        dataset_dir=Path('benchmarks'),
        output_dir=Path('results'),
        max_samples_per_task=10
    )
    await evaluator.close()
    print('Evaluation completed!')

asyncio.run(evaluate())
"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_services.py -v  # Service tests
pytest tests/test_agents.py -v    # Agent tests
pytest tests/test_pipeline.py -v  # Pipeline tests

# Run with coverage
pytest tests/ --cov=geodaedalus --cov-report=html
```

## 🔧 Configuration

GeoDaedalus uses Pydantic settings for configuration:

```python
from geodaedalus.core.config import Settings

settings = Settings(
    # API Keys
    openai_api_key="your-key",
    serpapi_key="your-key",
    
    # LLM Settings
    llm=Settings.LLMSettings(
        model="gpt-4",
        temperature=0.1,
        max_tokens=2000
    ),
    
    # Processing Settings
    processing=Settings.ProcessingSettings(
        max_file_size_mb=50,
        max_pdf_pages=100,
        max_concurrent_downloads=5
    ),
    
    # Search Settings
    search=Settings.SearchSettings(
        max_papers_per_query=50,
        min_citation_count=1,
        preferred_engines=["semantic_scholar", "google_scholar"]
    )
)
```

## 📋 Data Models

GeoDaedalus uses structured Pydantic models for all data:

```python
from geodaedalus.core.models import (
    GeoscientificConstraints,  # Query constraints
    LiteraturePaper,          # Paper metadata
    ExtractedData,            # Extracted datasets
    DataTable,                # Structured tables
    FusedData                 # Final results
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/geodaedalus.git
cd geodaedalus
pip install -e ".[dev]"
pre-commit install
```

### Adding New Features

1. **New Agents**: Extend `BaseAgent` class
2. **New Services**: Implement service interfaces
3. **New Evaluators**: Add task-specific evaluation logic
4. **New Data Types**: Extend Pydantic models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and API
- **Semantic Scholar** for academic paper access
- **SerpAPI** for Google Scholar integration
- **PyPDF** and **pdfminer** for PDF processing
- **Rich** for beautiful terminal UI
- **Pydantic** for data validation
- **FastAPI** for web API framework

## 📞 Support

- 📧 Email: geodaedalus@example.org
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/geodaedalus/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/geodaedalus/discussions)

---

**GeoDaedalus** - Navigating the maze of geoscientific literature with AI 🌍🔬✨

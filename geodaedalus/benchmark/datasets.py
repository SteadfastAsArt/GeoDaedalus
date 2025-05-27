"""Benchmark datasets for GeoDaedalus evaluation."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel

from geodaedalus.core.models import (
    GeoscientificConstraints,
    LiteraturePaper,
    DataTable,
    ExtractedData,
    SearchResults,
)


class BenchmarkTask(str, Enum):
    """Types of benchmark tasks."""
    
    INTENT_UNDERSTANDING = "intent_understanding"
    LITERATURE_RETRIEVAL = "literature_retrieval"
    DATA_EXTRACTION = "data_extraction"
    DATA_FUSION = "data_fusion"
    END_TO_END = "end_to_end"


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark samples."""
    
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class BenchmarkSample:
    """Single benchmark sample."""
    
    id: str
    task: BenchmarkTask
    difficulty: DifficultyLevel
    input_data: Any
    expected_output: Any
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task": self.task.value,
            "difficulty": self.difficulty.value,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            task=BenchmarkTask(data["task"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            input_data=data["input_data"],
            expected_output=data["expected_output"],
            metadata=data.get("metadata", {}),
        )


class BaseBenchmarkDataset(ABC):
    """Base class for benchmark datasets."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.samples: List[BenchmarkSample] = []
    
    @abstractmethod
    def generate_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """Generate benchmark samples."""
        pass
    
    def load_from_file(self, file_path: Path) -> None:
        """Load dataset from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.samples = [BenchmarkSample.from_dict(sample) for sample in data["samples"]]
    
    def save_to_file(self, file_path: Path) -> None:
        """Save dataset to file."""
        data = {
            "name": self.name,
            "description": self.description,
            "task": self.samples[0].task.value if self.samples else None,
            "sample_count": len(self.samples),
            "samples": [sample.to_dict() for sample in self.samples]
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_samples(
        self, 
        difficulty: Optional[DifficultyLevel] = None,
        count: Optional[int] = None
    ) -> List[BenchmarkSample]:
        """Get samples with optional filtering."""
        filtered_samples = self.samples
        
        if difficulty:
            filtered_samples = [s for s in filtered_samples if s.difficulty == difficulty]
        
        if count:
            filtered_samples = filtered_samples[:count]
        
        return filtered_samples


class IntentUnderstandingDataset(BaseBenchmarkDataset):
    """Dataset for testing requirement understanding capabilities."""
    
    def __init__(self):
        super().__init__(
            name="Intent Understanding",
            description="Tests the ability to parse natural language queries into structured geoscientific constraints"
        )
    
    def generate_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """Generate intent understanding samples."""
        samples = []
        
        # Template queries with expected outputs
        query_templates = [
            # Easy queries
            {
                "query": "Find igneous rocks from Hawaii",
                "expected": {
                    "spatial": {"location_name": "Hawaii", "country": "USA"},
                    "rock_types": ["igneous"],
                    "element_constraints": [],
                },
                "difficulty": DifficultyLevel.EASY
            },
            {
                "query": "Get volcanic data with major elements",
                "expected": {
                    "spatial": None,
                    "rock_types": ["volcanic"],
                    "element_constraints": [{"category": "major", "elements": ["SiO2", "Al2O3", "Fe2O3", "MgO", "CaO"]}],
                },
                "difficulty": DifficultyLevel.EASY
            },
            
            # Medium queries
            {
                "query": "Find Cretaceous basalts from the Deccan Traps with trace element concentrations",
                "expected": {
                    "spatial": {"location_name": "Deccan Traps", "country": "India"},
                    "temporal": {"geological_period": "Cretaceous", "age_min": 66, "age_max": 145},
                    "rock_types": ["basalt", "volcanic"],
                    "element_constraints": [{"category": "minor", "elements": ["Rb", "Sr", "Ba", "Y", "Zr"]}],
                },
                "difficulty": DifficultyLevel.MEDIUM
            },
            {
                "query": "Get sedimentary rocks from Permian period with carbonate compositions",
                "expected": {
                    "spatial": None,
                    "temporal": {"geological_period": "Permian", "age_min": 252, "age_max": 299},
                    "rock_types": ["sedimentary", "carbonate"],
                    "element_constraints": [{"category": "major", "elements": ["CaO", "MgO", "SiO2"]}],
                },
                "difficulty": DifficultyLevel.MEDIUM
            },
            
            # Hard queries
            {
                "query": "Find Archean metamorphic rocks from the Superior Province with isotope ratios and REE patterns excluding altered samples",
                "expected": {
                    "spatial": {"location_name": "Superior Province", "country": "Canada"},
                    "temporal": {"geological_period": "Archean", "age_min": 2500, "age_max": 4000},
                    "rock_types": ["metamorphic"],
                    "element_constraints": [
                        {"category": "isotopes", "elements": ["87Sr/86Sr", "143Nd/144Nd"]},
                        {"category": "minor", "elements": ["La", "Ce", "Nd", "Sm", "Eu"]}
                    ],
                    "additional_keywords": ["unaltered", "fresh"]
                },
                "difficulty": DifficultyLevel.HARD
            }
        ]
        
        # Generate samples from templates
        for i, template in enumerate(query_templates):
            for variant in range(count // len(query_templates)):
                sample = BenchmarkSample(
                    id=f"intent_{i}_{variant}",
                    task=BenchmarkTask.INTENT_UNDERSTANDING,
                    difficulty=template["difficulty"],
                    input_data={"user_query": template["query"]},
                    expected_output=template["expected"],
                    metadata={
                        "template_id": i,
                        "variant": variant,
                        "query_complexity": len(template["query"].split()),
                    }
                )
                samples.append(sample)
        
        self.samples = samples[:count]
        return self.samples


class LiteratureRetrievalDataset(BaseBenchmarkDataset):
    """Dataset for testing literature search capabilities."""
    
    def __init__(self):
        super().__init__(
            name="Literature Retrieval",
            description="Tests the ability to find relevant papers for geoscientific queries"
        )
    
    def generate_samples(self, count: int = 50) -> List[BenchmarkSample]:
        """Generate literature retrieval samples."""
        samples = []
        
        # Known paper references for validation
        reference_papers = [
            {
                "query": "Hawaii basalt geochemistry",
                "expected_papers": [
                    {"title": "Geochemical Evolution of Hawaiian Volcanic Series", "relevance": 0.95},
                    {"title": "Major Element Compositions of Hawaiian Basalts", "relevance": 0.90},
                ],
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "query": "Deccan Traps flood basalt mantle source",
                "expected_papers": [
                    {"title": "Mantle Sources of Deccan Volcanism", "relevance": 0.92},
                    {"title": "Geochemistry of Deccan Flood Basalts", "relevance": 0.88},
                ],
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "query": "Archean komatiite trace element petrogenesis",
                "expected_papers": [
                    {"title": "Trace Element Geochemistry of Archean Komatiites", "relevance": 0.90},
                    {"title": "Petrogenesis of Komatiites from Barberton", "relevance": 0.85},
                ],
                "difficulty": DifficultyLevel.HARD,
            }
        ]
        
        for i, ref in enumerate(reference_papers):
            sample = BenchmarkSample(
                id=f"retrieval_{i}",
                task=BenchmarkTask.LITERATURE_RETRIEVAL,
                difficulty=ref["difficulty"],
                input_data={"search_query": ref["query"]},
                expected_output={
                    "expected_papers": ref["expected_papers"],
                    "min_relevance": 0.7,
                    "min_papers": 5,
                },
                metadata={
                    "domain": "geochemistry",
                    "query_type": "academic_search",
                }
            )
            samples.append(sample)
        
        self.samples = samples
        return self.samples


class DataExtractionDataset(BaseBenchmarkDataset):
    """Dataset for testing data extraction capabilities."""
    
    def __init__(self):
        super().__init__(
            name="Data Extraction", 
            description="Tests the ability to extract structured data from literature"
        )
    
    def generate_samples(self, count: int = 30) -> List[BenchmarkSample]:
        """Generate data extraction samples."""
        samples = []
        
        # Mock paper content with expected extractions
        mock_papers = [
            {
                "content": """
                Table 1. Major element compositions of Hawaiian basalts (wt%)
                Sample    SiO2    Al2O3   Fe2O3   MgO     CaO     Na2O    K2O
                HAW-1     49.2    13.8    12.1    8.4     11.2    2.8     0.6
                HAW-2     47.8    14.2    13.2    9.1     10.8    2.4     0.5
                HAW-3     50.1    13.5    11.8    7.9     10.9    3.1     0.7
                """,
                "expected_tables": [
                    {
                        "headers": ["Sample", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO", "Na2O", "K2O"],
                        "data": [
                            ["HAW-1", "49.2", "13.8", "12.1", "8.4", "11.2", "2.8", "0.6"],
                            ["HAW-2", "47.8", "14.2", "13.2", "9.1", "10.8", "2.4", "0.5"],
                            ["HAW-3", "50.1", "13.5", "11.8", "7.9", "10.9", "3.1", "0.7"],
                        ],
                        "element_data": {
                            "SiO2": [49.2, 47.8, 50.1],
                            "Al2O3": [13.8, 14.2, 13.5],
                            "Fe2O3": [12.1, 13.2, 11.8],
                        }
                    }
                ],
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "content": """
                Trace element concentrations were determined by ICP-MS.
                Table 3. Trace element data for Deccan basalts (ppm)
                Sample  Location      Rb    Sr    Ba    Y     Zr    Nb
                DT-01   Western Ghats 15    380   250   28    110   8
                DT-02   Western Ghats 12    420   290   32    125   12
                DT-03   Central India 18    360   220   25    105   7
                """,
                "expected_tables": [
                    {
                        "headers": ["Sample", "Location", "Rb", "Sr", "Ba", "Y", "Zr", "Nb"],
                        "data": [
                            ["DT-01", "Western Ghats", "15", "380", "250", "28", "110", "8"],
                            ["DT-02", "Western Ghats", "12", "420", "290", "32", "125", "12"],
                            ["DT-03", "Central India", "18", "360", "220", "25", "105", "7"],
                        ],
                        "element_data": {
                            "Rb": [15, 12, 18],
                            "Sr": [380, 420, 360],
                            "Ba": [250, 290, 220],
                        }
                    }
                ],
                "difficulty": DifficultyLevel.MEDIUM,
            }
        ]
        
        # Generate samples by cycling through mock papers to reach the requested count
        for i in range(count):
            mock = mock_papers[i % len(mock_papers)]
            sample = BenchmarkSample(
                id=f"extraction_{i}",
                task=BenchmarkTask.DATA_EXTRACTION,
                difficulty=mock["difficulty"],
                input_data={
                    "paper_content": mock["content"],
                    "paper_id": f"mock_paper_{i}",
                },
                expected_output={
                    "tables_count": len(mock["expected_tables"]),
                    "expected_tables": mock["expected_tables"],
                },
                metadata={
                    "content_type": "table",
                    "element_categories": ["major", "minor"],
                }
            )
            samples.append(sample)
        
        self.samples = samples
        return self.samples


class DataFusionDataset(BaseBenchmarkDataset):
    """Dataset for testing data fusion capabilities."""
    
    def __init__(self):
        super().__init__(
            name="Data Fusion",
            description="Tests the ability to merge and validate data from multiple sources"
        )
    
    def generate_samples(self, count: int = 20) -> List[BenchmarkSample]:
        """Generate data fusion samples."""
        samples = []
        
        # Mock multi-source data scenarios
        fusion_scenarios = [
            {
                "scenario": "duplicate_removal",
                "input_tables": [
                    # Table from Paper 1
                    {
                        "paper_id": "paper_1",
                        "headers": ["Sample", "SiO2", "Al2O3", "MgO"],
                        "data": [["S1", "49.2", "13.8", "8.4"], ["S2", "47.8", "14.2", "9.1"]],
                        "confidence": 0.9,
                    },
                    # Duplicate table from Paper 2 (same data, different format)
                    {
                        "paper_id": "paper_2", 
                        "headers": ["Sample ID", "SiO2 (wt%)", "Al2O3 (wt%)", "MgO (wt%)"],
                        "data": [["S1", "49.2", "13.8", "8.4"], ["S2", "47.8", "14.2", "9.1"]],
                        "confidence": 0.8,
                    },
                ],
                "expected_output": {
                    "final_tables_count": 1,  # Should merge duplicates
                    "quality_score": 0.9,
                    "data_completeness": 1.0,
                },
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "scenario": "outlier_detection",
                "input_tables": [
                    {
                        "paper_id": "paper_1",
                        "headers": ["Sample", "SiO2", "Al2O3"],
                        "data": [["S1", "49.2", "13.8"], ["S2", "47.8", "14.2"], ["S3", "95.0", "2.1"]],  # S3 is outlier
                        "confidence": 0.9,
                    },
                ],
                "expected_output": {
                    "outliers_detected": 1,
                    "outlier_samples": ["S3"],
                    "quality_score": 0.75,
                },
                "difficulty": DifficultyLevel.MEDIUM,
            }
        ]
        
        for i, scenario in enumerate(fusion_scenarios):
            sample = BenchmarkSample(
                id=f"fusion_{i}",
                task=BenchmarkTask.DATA_FUSION,
                difficulty=scenario["difficulty"],
                input_data={
                    "scenario": scenario["scenario"],
                    "input_tables": scenario["input_tables"],
                },
                expected_output=scenario["expected_output"],
                metadata={
                    "fusion_type": scenario["scenario"],
                    "complexity": len(scenario["input_tables"]),
                }
            )
            samples.append(sample)
        
        self.samples = samples
        return self.samples


class GeoDataBenchDataset:
    """Main benchmark dataset collection for GeoDaedalus."""
    
    def __init__(self):
        self.datasets = {
            BenchmarkTask.INTENT_UNDERSTANDING: IntentUnderstandingDataset(),
            BenchmarkTask.LITERATURE_RETRIEVAL: LiteratureRetrievalDataset(),
            BenchmarkTask.DATA_EXTRACTION: DataExtractionDataset(),
            BenchmarkTask.DATA_FUSION: DataFusionDataset(),
        }
    
    def get_dataset(self, task: BenchmarkTask) -> BaseBenchmarkDataset:
        """Get dataset for specific task."""
        return self.datasets[task]
    
    def generate_all_datasets(self, base_dir: Path) -> None:
        """Generate and save all benchmark datasets."""
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for task, dataset in self.datasets.items():
            # Generate samples based on task complexity
            sample_counts = {
                BenchmarkTask.INTENT_UNDERSTANDING: 100,
                BenchmarkTask.LITERATURE_RETRIEVAL: 50,
                BenchmarkTask.DATA_EXTRACTION: 30,
                BenchmarkTask.DATA_FUSION: 20,
            }
            
            dataset.generate_samples(sample_counts[task])
            
            # Save to file
            file_path = base_dir / f"{task.value}_dataset.json"
            dataset.save_to_file(file_path)
            
            print(f"Generated {len(dataset.samples)} samples for {task.value}")
    
    def load_all_datasets(self, base_dir: Path) -> None:
        """Load all datasets from files."""
        for task, dataset in self.datasets.items():
            file_path = base_dir / f"{task.value}_dataset.json"
            if file_path.exists():
                dataset.load_from_file(file_path)
                print(f"Loaded {len(dataset.samples)} samples for {task.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all datasets."""
        stats = {
            "total_samples": 0,
            "tasks": {},
            "difficulty_distribution": {level.value: 0 for level in DifficultyLevel},
        }
        
        for task, dataset in self.datasets.items():
            task_stats = {
                "sample_count": len(dataset.samples),
                "difficulty_breakdown": {level.value: 0 for level in DifficultyLevel},
            }
            
            for sample in dataset.samples:
                task_stats["difficulty_breakdown"][sample.difficulty.value] += 1
                stats["difficulty_distribution"][sample.difficulty.value] += 1
            
            stats["tasks"][task.value] = task_stats
            stats["total_samples"] += len(dataset.samples)
        
        return stats 
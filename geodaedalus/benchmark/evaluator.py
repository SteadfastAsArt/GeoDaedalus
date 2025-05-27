"""Benchmark evaluation system for GeoDaedalus."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from geodaedalus.benchmark.datasets import (
    BenchmarkSample,
    BenchmarkTask,
    DifficultyLevel,
    GeoDataBenchDataset,
)
from geodaedalus.core.config import Settings
from geodaedalus.core.logging import get_logger
from geodaedalus.core.pipeline import GeoDaedalusPipeline

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    execution_time: float = 0.0
    cost_estimate: float = 0.0
    error_rate: float = 0.0
    completeness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "execution_time": self.execution_time,
            "cost_estimate": self.cost_estimate,
            "error_rate": self.error_rate,
            "completeness_score": self.completeness_score,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a single benchmark sample."""
    
    sample_id: str
    task: BenchmarkTask
    difficulty: DifficultyLevel
    predicted_output: Any
    expected_output: Any
    metrics: EvaluationMetrics
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "task": self.task.value,
            "difficulty": self.difficulty.value,
            "predicted_output": self.predicted_output,
            "expected_output": self.expected_output,
            "metrics": self.metrics.to_dict(),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class TaskEvaluator:
    """Base class for task-specific evaluators."""
    
    def __init__(self, task: BenchmarkTask):
        self.task = task
    
    def evaluate_sample(
        self, 
        predicted: Any, 
        expected: Any, 
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluate a single sample."""
        raise NotImplementedError
    
    def calculate_accuracy(self, predicted: Any, expected: Any) -> float:
        """Calculate accuracy score."""
        raise NotImplementedError


class IntentUnderstandingEvaluator(TaskEvaluator):
    """Evaluator for intent understanding task."""
    
    def __init__(self):
        super().__init__(BenchmarkTask.INTENT_UNDERSTANDING)
    
    def evaluate_sample(
        self, 
        predicted: Any, 
        expected: Any, 
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluate intent understanding prediction."""
        metrics = EvaluationMetrics(execution_time=execution_time)
        
        try:
            # Convert predicted constraints to comparable format
            pred_dict = self._constraints_to_dict(predicted)
            exp_dict = expected
            
            # Calculate component accuracies
            spatial_acc = self._evaluate_spatial(pred_dict.get("spatial"), exp_dict.get("spatial"))
            temporal_acc = self._evaluate_temporal(pred_dict.get("temporal"), exp_dict.get("temporal"))
            rock_types_acc = self._evaluate_rock_types(pred_dict.get("rock_types", []), exp_dict.get("rock_types", []))
            elements_acc = self._evaluate_elements(pred_dict.get("element_constraints", []), exp_dict.get("element_constraints", []))
            
            # Overall accuracy
            metrics.accuracy = (spatial_acc + temporal_acc + rock_types_acc + elements_acc) / 4
            
            # Precision and recall for extractable fields
            metrics.precision, metrics.recall = self._calculate_precision_recall(pred_dict, exp_dict)
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0.0
            
            # Completeness (how much of expected output was captured)
            metrics.completeness_score = self._calculate_completeness(pred_dict, exp_dict)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            metrics.error_rate = 1.0
        
        return metrics
    
    def _constraints_to_dict(self, constraints) -> Dict[str, Any]:
        """Convert GeoscientificConstraints to dictionary."""
        if hasattr(constraints, 'dict'):
            return constraints.dict()
        elif isinstance(constraints, dict):
            return constraints
        else:
            return {}
    
    def _evaluate_spatial(self, predicted: Optional[Dict], expected: Optional[Dict]) -> float:
        """Evaluate spatial constraints."""
        if not expected:
            return 1.0 if not predicted else 0.8  # Slight penalty for over-extraction
        if not predicted:
            return 0.0
        
        score = 0.0
        total_fields = 0
        
        for field in ["location_name", "country", "region"]:
            if field in expected:
                total_fields += 1
                pred_val = predicted.get(field, "").lower() if predicted.get(field) else ""
                exp_val = expected.get(field, "").lower() if expected.get(field) else ""
                
                if exp_val in pred_val or pred_val in exp_val:
                    score += 1
        
        return score / total_fields if total_fields > 0 else 1.0
    
    def _evaluate_temporal(self, predicted: Optional[Dict], expected: Optional[Dict]) -> float:
        """Evaluate temporal constraints."""
        if not expected:
            return 1.0 if not predicted else 0.8
        if not predicted:
            return 0.0
        
        score = 0.0
        total_fields = 0
        
        # Check geological period
        if expected.get("geological_period"):
            total_fields += 1
            pred_period = predicted.get("geological_period", "").lower()
            exp_period = expected.get("geological_period", "").lower()
            if exp_period in pred_period or pred_period in exp_period:
                score += 1
        
        # Check age ranges (allow some tolerance)
        for age_field in ["age_min", "age_max"]:
            if expected.get(age_field) is not None:
                total_fields += 1
                pred_age = predicted.get(age_field)
                exp_age = expected.get(age_field)
                if pred_age is not None and abs(pred_age - exp_age) <= exp_age * 0.1:  # 10% tolerance
                    score += 1
        
        return score / total_fields if total_fields > 0 else 1.0
    
    def _evaluate_rock_types(self, predicted: List[str], expected: List[str]) -> float:
        """Evaluate rock types."""
        if not expected:
            return 1.0 if not predicted else 0.8
        if not predicted:
            return 0.0
        
        pred_set = {rt.lower() for rt in predicted}
        exp_set = {rt.lower() for rt in expected}
        
        # Intersection over union
        intersection = len(pred_set.intersection(exp_set))
        union = len(pred_set.union(exp_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _evaluate_elements(self, predicted: List[Dict], expected: List[Dict]) -> float:
        """Evaluate element constraints."""
        if not expected:
            return 1.0 if not predicted else 0.8
        if not predicted:
            return 0.0
        
        score = 0.0
        
        for exp_constraint in expected:
            exp_category = exp_constraint.get("category", "").lower()
            exp_elements = {e.lower() for e in exp_constraint.get("elements", [])}
            
            # Find matching category in predicted
            best_match_score = 0.0
            for pred_constraint in predicted:
                pred_category = pred_constraint.get("category", "").lower()
                if pred_category == exp_category:
                    pred_elements = {e.lower() for e in pred_constraint.get("elements", [])}
                    
                    # Calculate element overlap
                    if exp_elements and pred_elements:
                        intersection = len(exp_elements.intersection(pred_elements))
                        union = len(exp_elements.union(pred_elements))
                        overlap_score = intersection / union if union > 0 else 0.0
                        best_match_score = max(best_match_score, overlap_score)
            
            score += best_match_score
        
        return score / len(expected) if expected else 1.0
    
    def _calculate_precision_recall(self, predicted: Dict, expected: Dict) -> Tuple[float, float]:
        """Calculate precision and recall."""
        # Count extracted vs expected fields
        extracted_fields = 0
        correct_fields = 0
        total_expected = 0
        
        # Count spatial fields
        if expected.get("spatial"):
            for field in ["location_name", "country", "region"]:
                if expected["spatial"].get(field):
                    total_expected += 1
                    if predicted.get("spatial", {}).get(field):
                        extracted_fields += 1
                        # Simple string matching for correctness
                        if field in predicted["spatial"] and field in expected["spatial"]:
                            correct_fields += 1
        
        # Count temporal fields
        if expected.get("temporal"):
            for field in ["geological_period", "age_min", "age_max"]:
                if expected["temporal"].get(field) is not None:
                    total_expected += 1
                    if predicted.get("temporal", {}).get(field) is not None:
                        extracted_fields += 1
                        correct_fields += 1  # Simplified
        
        # Count rock types
        if expected.get("rock_types"):
            total_expected += len(expected["rock_types"])
            extracted_fields += len(predicted.get("rock_types", []))
            # Count overlapping rock types
            pred_rocks = {rt.lower() for rt in predicted.get("rock_types", [])}
            exp_rocks = {rt.lower() for rt in expected["rock_types"]}
            correct_fields += len(pred_rocks.intersection(exp_rocks))
        
        precision = correct_fields / extracted_fields if extracted_fields > 0 else 0.0
        recall = correct_fields / total_expected if total_expected > 0 else 0.0
        
        return precision, recall
    
    def _calculate_completeness(self, predicted: Dict, expected: Dict) -> float:
        """Calculate how complete the prediction is."""
        total_components = 0
        completed_components = 0
        
        # Check each major component
        components = ["spatial", "temporal", "rock_types", "element_constraints"]
        
        for component in components:
            if expected.get(component):
                total_components += 1
                if predicted.get(component):
                    completed_components += 1
        
        return completed_components / total_components if total_components > 0 else 1.0


class LiteratureRetrievalEvaluator(TaskEvaluator):
    """Evaluator for literature retrieval task."""
    
    def __init__(self):
        super().__init__(BenchmarkTask.LITERATURE_RETRIEVAL)
    
    def evaluate_sample(
        self, 
        predicted: Any, 
        expected: Any, 
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluate literature retrieval results."""
        metrics = EvaluationMetrics(execution_time=execution_time)
        
        try:
            predicted_papers = predicted if isinstance(predicted, list) else []
            expected_config = expected
            
            # Check minimum number of papers
            min_papers = expected_config.get("min_papers", 5)
            metrics.completeness_score = min(len(predicted_papers) / min_papers, 1.0)
            
            # Calculate relevance scores
            if predicted_papers and "expected_papers" in expected_config:
                relevance_scores = self._calculate_relevance_scores(
                    predicted_papers, 
                    expected_config["expected_papers"]
                )
                metrics.accuracy = sum(relevance_scores) / len(relevance_scores)
                
                # Precision: relevant papers / total retrieved
                min_relevance = expected_config.get("min_relevance", 0.7)
                relevant_count = sum(1 for score in relevance_scores if score >= min_relevance)
                metrics.precision = relevant_count / len(predicted_papers) if predicted_papers else 0.0
                
                # Recall: found expected papers / total expected
                expected_count = len(expected_config["expected_papers"])
                metrics.recall = relevant_count / expected_count if expected_count > 0 else 0.0
                
                if metrics.precision + metrics.recall > 0:
                    metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            
        except Exception as e:
            logger.error(f"Literature retrieval evaluation failed: {e}")
            metrics.error_rate = 1.0
        
        return metrics
    
    def _calculate_relevance_scores(self, predicted_papers: List[Any], expected_papers: List[Dict]) -> List[float]:
        """Calculate relevance scores by comparing titles."""
        scores = []
        
        for pred_paper in predicted_papers:
            pred_title = getattr(pred_paper, 'title', '').lower()
            best_score = 0.0
            
            for exp_paper in expected_papers:
                exp_title = exp_paper.get('title', '').lower()
                # Simple title similarity
                overlap = self._title_similarity(pred_title, exp_title)
                best_score = max(best_score, overlap)
            
            scores.append(best_score)
        
        return scores
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity."""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union


class DataExtractionEvaluator(TaskEvaluator):
    """Evaluator for data extraction task."""
    
    def __init__(self):
        super().__init__(BenchmarkTask.DATA_EXTRACTION)
    
    def evaluate_sample(
        self, 
        predicted: Any, 
        expected: Any, 
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluate data extraction results."""
        metrics = EvaluationMetrics(execution_time=execution_time)
        
        try:
            pred_result = predicted if isinstance(predicted, dict) else {}
            exp_result = expected
            
            # Check table count
            pred_tables = pred_result.get("tables", [])
            exp_tables_count = exp_result.get("tables_count", 0)
            
            if exp_tables_count > 0:
                metrics.completeness_score = min(len(pred_tables) / exp_tables_count, 1.0)
            
            # Evaluate table structure and content
            if pred_tables and "expected_tables" in exp_result:
                table_scores = []
                
                for exp_table in exp_result["expected_tables"]:
                    best_score = 0.0
                    
                    for pred_table in pred_tables:
                        score = self._evaluate_table_match(pred_table, exp_table)
                        best_score = max(best_score, score)
                    
                    table_scores.append(best_score)
                
                metrics.accuracy = sum(table_scores) / len(table_scores) if table_scores else 0.0
                metrics.precision = metrics.accuracy  # Simplified
                metrics.recall = metrics.accuracy
                metrics.f1_score = metrics.accuracy
            
        except Exception as e:
            logger.error(f"Data extraction evaluation failed: {e}")
            metrics.error_rate = 1.0
        
        return metrics
    
    def _evaluate_table_match(self, predicted_table: Dict, expected_table: Dict) -> float:
        """Evaluate how well a predicted table matches expected."""
        score = 0.0
        total_components = 0
        
        # Check headers
        if "headers" in expected_table:
            total_components += 1
            pred_headers = predicted_table.get("headers", [])
            exp_headers = expected_table["headers"]
            
            if pred_headers and exp_headers:
                header_overlap = len(set(pred_headers).intersection(set(exp_headers)))
                header_score = header_overlap / len(exp_headers)
                score += header_score
        
        # Check data rows count
        if "data" in expected_table:
            total_components += 1
            pred_data = predicted_table.get("data", [])
            exp_data = expected_table["data"]
            
            if pred_data and exp_data:
                row_count_score = min(len(pred_data) / len(exp_data), 1.0)
                score += row_count_score
        
        # Check element data if available
        if "element_data" in expected_table:
            total_components += 1
            pred_elements = predicted_table.get("element_data", {})
            exp_elements = expected_table["element_data"]
            
            if pred_elements and exp_elements:
                element_overlap = len(set(pred_elements.keys()).intersection(set(exp_elements.keys())))
                element_score = element_overlap / len(exp_elements) if exp_elements else 0.0
                score += element_score
        
        return score / total_components if total_components > 0 else 0.0


class DataFusionEvaluator(TaskEvaluator):
    """Evaluator for data fusion task."""
    
    def __init__(self):
        super().__init__(BenchmarkTask.DATA_FUSION)
    
    def evaluate_sample(
        self, 
        predicted: Any, 
        expected: Any, 
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluate data fusion results."""
        metrics = EvaluationMetrics(execution_time=execution_time)
        
        try:
            pred_result = predicted if isinstance(predicted, dict) else {}
            exp_result = expected
            
            # Check fusion outcomes
            accuracy_components = []
            
            # Check table count (for duplicate removal)
            if "final_tables_count" in exp_result:
                pred_count = pred_result.get("final_tables_count", 0)
                exp_count = exp_result["final_tables_count"]
                count_accuracy = 1.0 if pred_count == exp_count else 0.0
                accuracy_components.append(count_accuracy)
            
            # Check outlier detection
            if "outliers_detected" in exp_result:
                pred_outliers = pred_result.get("outliers_detected", 0)
                exp_outliers = exp_result["outliers_detected"]
                outlier_accuracy = 1.0 if pred_outliers == exp_outliers else 0.0
                accuracy_components.append(outlier_accuracy)
            
            # Check quality scores
            if "quality_score" in exp_result:
                pred_quality = pred_result.get("quality_score", 0.0)
                exp_quality = exp_result["quality_score"]
                quality_diff = abs(pred_quality - exp_quality)
                quality_accuracy = max(0.0, 1.0 - quality_diff)
                accuracy_components.append(quality_accuracy)
            
            metrics.accuracy = sum(accuracy_components) / len(accuracy_components) if accuracy_components else 0.0
            metrics.precision = metrics.accuracy
            metrics.recall = metrics.accuracy
            metrics.f1_score = metrics.accuracy
            
            # Completeness based on having fusion results
            metrics.completeness_score = 1.0 if pred_result else 0.0
            
        except Exception as e:
            logger.error(f"Data fusion evaluation failed: {e}")
            metrics.error_rate = 1.0
        
        return metrics


class BenchmarkEvaluator:
    """Main benchmark evaluation system."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()
        
        # Task-specific evaluators
        self.evaluators = {
            BenchmarkTask.INTENT_UNDERSTANDING: IntentUnderstandingEvaluator(),
            BenchmarkTask.LITERATURE_RETRIEVAL: LiteratureRetrievalEvaluator(),
            BenchmarkTask.DATA_EXTRACTION: DataExtractionEvaluator(),
            BenchmarkTask.DATA_FUSION: DataFusionEvaluator(),
        }
        
        # Initialize pipeline for end-to-end evaluation
        self.pipeline = GeoDaedalusPipeline(settings)
    
    async def evaluate_agent(
        self, 
        agent: Any, 
        dataset: List[BenchmarkSample],
        max_samples: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Evaluate a single agent on a dataset."""
        results = []
        
        samples_to_eval = dataset[:max_samples] if max_samples else dataset
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Evaluating {agent.agent_name}...", total=len(samples_to_eval))
            
            for sample in samples_to_eval:
                try:
                    start_time = time.time()
                    
                    # Execute agent
                    predicted_output = await agent.process(sample.input_data)
                    
                    execution_time = time.time() - start_time
                    
                    # Evaluate result
                    evaluator = self.evaluators[sample.task]
                    metrics = evaluator.evaluate_sample(
                        predicted_output, 
                        sample.expected_output, 
                        execution_time
                    )
                    
                    result = EvaluationResult(
                        sample_id=sample.id,
                        task=sample.task,
                        difficulty=sample.difficulty,
                        predicted_output=predicted_output,
                        expected_output=sample.expected_output,
                        metrics=metrics
                    )
                    
                except Exception as e:
                    logger.error(f"Agent evaluation failed for sample {sample.id}: {e}")
                    result = EvaluationResult(
                        sample_id=sample.id,
                        task=sample.task,
                        difficulty=sample.difficulty,
                        predicted_output=None,
                        expected_output=sample.expected_output,
                        metrics=EvaluationMetrics(error_rate=1.0),
                        error_message=str(e)
                    )
                
                results.append(result)
                progress.advance(task)
        
        return results
    
    async def evaluate_pipeline_end_to_end(
        self, 
        dataset: List[BenchmarkSample],
        max_samples: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Evaluate the complete pipeline end-to-end."""
        results = []
        
        samples_to_eval = dataset[:max_samples] if max_samples else dataset
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Evaluating end-to-end pipeline...", total=len(samples_to_eval))
            
            for sample in samples_to_eval:
                try:
                    start_time = time.time()
                    
                    # Extract user query
                    user_query = sample.input_data.get("user_query", "")
                    if not user_query:
                        continue
                    
                    # Run complete pipeline
                    pipeline_result = await self.pipeline.run_complete_pipeline(
                        user_query=user_query,
                        max_papers=10
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Evaluate final result
                    # For end-to-end, we evaluate the quality of extracted data
                    metrics = self._evaluate_end_to_end_result(
                        pipeline_result, 
                        sample.expected_output, 
                        execution_time
                    )
                    
                    result = EvaluationResult(
                        sample_id=sample.id,
                        task=BenchmarkTask.END_TO_END,
                        difficulty=sample.difficulty,
                        predicted_output=pipeline_result,
                        expected_output=sample.expected_output,
                        metrics=metrics
                    )
                    
                except Exception as e:
                    logger.error(f"Pipeline evaluation failed for sample {sample.id}: {e}")
                    result = EvaluationResult(
                        sample_id=sample.id,
                        task=BenchmarkTask.END_TO_END,
                        difficulty=sample.difficulty,
                        predicted_output=None,
                        expected_output=sample.expected_output,
                        metrics=EvaluationMetrics(error_rate=1.0),
                        error_message=str(e)
                    )
                
                results.append(result)
                progress.advance(task)
        
        return results
    
    def _evaluate_end_to_end_result(
        self, 
        pipeline_result: Dict, 
        expected: Dict, 
        execution_time: float
    ) -> EvaluationMetrics:
        """Evaluate end-to-end pipeline result."""
        metrics = EvaluationMetrics(execution_time=execution_time)
        
        try:
            # Check if pipeline produced results
            has_papers = len(pipeline_result.get("papers", [])) > 0
            has_data = len(pipeline_result.get("extracted_data", [])) > 0
            has_fused_data = len(pipeline_result.get("fused_data", [])) > 0
            
            # Completeness score
            completion_components = [has_papers, has_data, has_fused_data]
            metrics.completeness_score = sum(completion_components) / len(completion_components)
            
            # Simplified accuracy based on having reasonable outputs
            metrics.accuracy = metrics.completeness_score
            
            # Quality assessment
            if has_fused_data:
                quality_metrics = pipeline_result.get("quality_metrics", {})
                avg_confidence = quality_metrics.get("average_confidence", 0.0)
                data_completeness = quality_metrics.get("data_completeness", 0.0)
                
                metrics.precision = avg_confidence
                metrics.recall = data_completeness
                
                if metrics.precision + metrics.recall > 0:
                    metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            
            # Cost estimation (based on execution time and API calls)
            estimated_api_calls = len(pipeline_result.get("papers", [])) * 2  # Rough estimate
            metrics.cost_estimate = estimated_api_calls * 0.002  # Rough cost per call
            
        except Exception as e:
            logger.error(f"End-to-end evaluation failed: {e}")
            metrics.error_rate = 1.0
        
        return metrics
    
    def generate_report(self, results: List[EvaluationResult], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate evaluation report."""
        if not results:
            return {}
        
        # Group results by task and difficulty
        task_results = {}
        difficulty_results = {}
        
        for result in results:
            task_key = result.task.value
            difficulty_key = result.difficulty.value
            
            if task_key not in task_results:
                task_results[task_key] = []
            if difficulty_key not in difficulty_results:
                difficulty_results[difficulty_key] = []
            
            task_results[task_key].append(result)
            difficulty_results[difficulty_key].append(result)
        
        # Calculate aggregate metrics
        report = {
            "summary": {
                "total_samples": len(results),
                "tasks_evaluated": list(task_results.keys()),
                "difficulty_levels": list(difficulty_results.keys()),
            },
            "overall_metrics": self._calculate_aggregate_metrics(results),
            "task_breakdown": {},
            "difficulty_breakdown": {},
            "detailed_results": []
        }
        
        # Task-level metrics
        for task, task_results_list in task_results.items():
            report["task_breakdown"][task] = self._calculate_aggregate_metrics(task_results_list)
        
        # Difficulty-level metrics
        for difficulty, diff_results_list in difficulty_results.items():
            report["difficulty_breakdown"][difficulty] = self._calculate_aggregate_metrics(diff_results_list)
        
        # Detailed results
        report["detailed_results"] = [result.to_dict() for result in results]
        
        # Save report
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        # Display summary
        self._display_report_summary(report)
        
        return report
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics for a list of results."""
        if not results:
            return {}
        
        metrics = {
            "accuracy": sum(r.metrics.accuracy for r in results) / len(results),
            "precision": sum(r.metrics.precision for r in results) / len(results),
            "recall": sum(r.metrics.recall for r in results) / len(results),
            "f1_score": sum(r.metrics.f1_score for r in results) / len(results),
            "avg_execution_time": sum(r.metrics.execution_time for r in results) / len(results),
            "total_cost_estimate": sum(r.metrics.cost_estimate for r in results),
            "error_rate": sum(r.metrics.error_rate for r in results) / len(results),
            "completeness_score": sum(r.metrics.completeness_score for r in results) / len(results),
        }
        
        return metrics
    
    def _display_report_summary(self, report: Dict[str, Any]) -> None:
        """Display report summary in console."""
        self.console.print("\n[bold blue]Evaluation Report Summary[/bold blue]")
        
        # Overall metrics table
        overall_table = Table(title="Overall Performance")
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="green")
        
        overall_metrics = report["overall_metrics"]
        for metric, value in overall_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            overall_table.add_row(metric.replace('_', ' ').title(), formatted_value)
        
        self.console.print(overall_table)
        
        # Task breakdown
        if report["task_breakdown"]:
            task_table = Table(title="Performance by Task")
            task_table.add_column("Task", style="cyan")
            task_table.add_column("Accuracy", style="green")
            task_table.add_column("F1 Score", style="green")
            task_table.add_column("Completeness", style="green")
            
            for task, metrics in report["task_breakdown"].items():
                task_table.add_row(
                    task.replace('_', ' ').title(),
                    f"{metrics.get('accuracy', 0):.3f}",
                    f"{metrics.get('f1_score', 0):.3f}",
                    f"{metrics.get('completeness_score', 0):.3f}"
                )
            
            self.console.print(task_table)
    
    async def run_comprehensive_evaluation(
        self, 
        dataset_dir: Path,
        output_dir: Path,
        max_samples_per_task: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on all tasks."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        benchmark_data = GeoDataBenchDataset()
        benchmark_data.load_all_datasets(dataset_dir)
        
        all_results = []
        
        # Evaluate each task
        for task in BenchmarkTask:
            if task == BenchmarkTask.END_TO_END:
                continue  # Handle separately
            
            self.console.print(f"\n[bold yellow]Evaluating {task.value}...[/bold yellow]")
            
            dataset = benchmark_data.get_dataset(task)
            samples = dataset.get_samples(count=max_samples_per_task)
            
            if not samples:
                continue
            
            # Get appropriate agent
            agent = await self._get_agent_for_task(task)
            if not agent:
                continue
            
            # Evaluate agent
            task_results = await self.evaluate_agent(agent, samples)
            all_results.extend(task_results)
            
            # Generate task-specific report
            task_report = self.generate_report(
                task_results, 
                output_dir / f"{task.value}_evaluation.json"
            )
        
        # Generate overall report
        overall_report = self.generate_report(
            all_results,
            output_dir / "overall_evaluation.json"
        )
        
        return overall_report
    
    async def _get_agent_for_task(self, task: BenchmarkTask):
        """Get the appropriate agent for a task."""
        if task == BenchmarkTask.INTENT_UNDERSTANDING:
            return self.pipeline.requirement_agent
        elif task == BenchmarkTask.LITERATURE_RETRIEVAL:
            return self.pipeline.literature_agent
        elif task == BenchmarkTask.DATA_EXTRACTION:
            return self.pipeline.extraction_agent
        elif task == BenchmarkTask.DATA_FUSION:
            return self.pipeline.fusion_agent
        else:
            return None
    
    async def close(self) -> None:
        """Close the evaluator."""
        await self.pipeline.close() 
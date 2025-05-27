"""Benchmark metrics and evaluation utilities for GeoDaedalus."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import numpy as np
from enum import Enum


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    EXECUTION_TIME = "execution_time"
    TOKEN_USAGE = "token_usage"
    COST = "cost"


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics."""
    
    # Core performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Text generation metrics
    bleu_score: Optional[float] = None
    rouge_l: Optional[float] = None
    semantic_similarity: Optional[float] = None
    
    # Efficiency metrics
    execution_time: Optional[float] = None
    token_usage: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    # Task-specific metrics
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize custom metrics dict if None."""
        if self.custom_metrics is None:
            self.custom_metrics = {}
    
    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {}
        
        # Core metrics
        if self.accuracy is not None:
            summary["accuracy"] = self.accuracy
        if self.precision is not None:
            summary["precision"] = self.precision
        if self.recall is not None:
            summary["recall"] = self.recall
        if self.f1_score is not None:
            summary["f1_score"] = self.f1_score
        
        # Text metrics
        if self.bleu_score is not None:
            summary["bleu_score"] = self.bleu_score
        if self.rouge_l is not None:
            summary["rouge_l"] = self.rouge_l
        if self.semantic_similarity is not None:
            summary["semantic_similarity"] = self.semantic_similarity
        
        # Efficiency metrics
        if self.execution_time is not None:
            summary["execution_time"] = self.execution_time
        if self.token_usage is not None:
            summary["token_usage"] = self.token_usage
        if self.cost_estimate is not None:
            summary["cost_estimate"] = self.cost_estimate
        
        # Custom metrics
        summary.update(self.custom_metrics)
        
        return summary


def calculate_accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
    """Calculate accuracy score."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if not predictions:
        return 0.0
    
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)


def calculate_precision_recall_f1(
    predictions: List[Any], 
    ground_truth: List[Any],
    positive_class: Any = True
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if not predictions:
        return 0.0, 0.0, 0.0
    
    # Convert to binary classification
    pred_positive = [p == positive_class for p in predictions]
    true_positive = [gt == positive_class for gt in ground_truth]
    
    tp = sum(1 for p, t in zip(pred_positive, true_positive) if p and t)
    fp = sum(1 for p, t in zip(pred_positive, true_positive) if p and not t)
    fn = sum(1 for p, t in zip(pred_positive, true_positive) if not p and t)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score for text generation."""
    # Simplified BLEU implementation
    if not predictions or not references:
        return 0.0
    
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        if not pred_words or not ref_words:
            continue
        
        # Calculate n-gram overlap (simplified)
        common_words = set(pred_words) & set(ref_words)
        score = len(common_words) / len(set(pred_words)) if pred_words else 0.0
        total_score += score
    
    return total_score / len(predictions) if predictions else 0.0


def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Calculate ROUGE-L score for text generation."""
    if not predictions or not references:
        return 0.0
    
    def lcs_length(s1: str, s2: str) -> int:
        """Calculate longest common subsequence length."""
        words1 = s1.lower().split()
        words2 = s2.lower().split()
        
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        lcs_len = lcs_length(pred, ref)
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        
        if pred_len == 0 or ref_len == 0:
            continue
        
        precision = lcs_len / pred_len
        recall = lcs_len / ref_len
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_score += f1
    
    return total_score / len(predictions) if predictions else 0.0


def calculate_semantic_similarity(
    predictions: List[str], 
    references: List[str],
    method: str = "jaccard"
) -> float:
    """Calculate semantic similarity between predictions and references."""
    if not predictions or not references:
        return 0.0
    
    if method == "jaccard":
        total_similarity = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            
            similarity = intersection / union if union > 0 else 0.0
            total_similarity += similarity
        
        return total_similarity / len(predictions)
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def aggregate_metrics(metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Aggregate multiple benchmark metrics into a single summary."""
    if not metrics_list:
        return BenchmarkMetrics()
    
    # Collect all metric values
    accuracies = [m.accuracy for m in metrics_list if m.accuracy is not None]
    precisions = [m.precision for m in metrics_list if m.precision is not None]
    recalls = [m.recall for m in metrics_list if m.recall is not None]
    f1_scores = [m.f1_score for m in metrics_list if m.f1_score is not None]
    bleu_scores = [m.bleu_score for m in metrics_list if m.bleu_score is not None]
    rouge_scores = [m.rouge_l for m in metrics_list if m.rouge_l is not None]
    semantic_sims = [m.semantic_similarity for m in metrics_list if m.semantic_similarity is not None]
    exec_times = [m.execution_time for m in metrics_list if m.execution_time is not None]
    token_usages = [m.token_usage for m in metrics_list if m.token_usage is not None]
    costs = [m.cost_estimate for m in metrics_list if m.cost_estimate is not None]
    
    # Calculate aggregated metrics
    aggregated = BenchmarkMetrics(
        accuracy=np.mean(accuracies) if accuracies else None,
        precision=np.mean(precisions) if precisions else None,
        recall=np.mean(recalls) if recalls else None,
        f1_score=np.mean(f1_scores) if f1_scores else None,
        bleu_score=np.mean(bleu_scores) if bleu_scores else None,
        rouge_l=np.mean(rouge_scores) if rouge_scores else None,
        semantic_similarity=np.mean(semantic_sims) if semantic_sims else None,
        execution_time=np.mean(exec_times) if exec_times else None,
        token_usage=int(np.sum(token_usages)) if token_usages else None,
        cost_estimate=np.sum(costs) if costs else None,
    )
    
    # Aggregate custom metrics
    all_custom_keys = set()
    for m in metrics_list:
        if m.custom_metrics:
            all_custom_keys.update(m.custom_metrics.keys())
    
    for key in all_custom_keys:
        values = [m.custom_metrics.get(key) for m in metrics_list if m.custom_metrics and key in m.custom_metrics]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if numeric_values:
            aggregated.add_custom_metric(key, np.mean(numeric_values))
    
    return aggregated 
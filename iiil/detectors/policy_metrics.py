import torch
import numpy as np
from typing import Callable, Dict


# Distance metrics inspired by the reference implementation
# Implemented such that always positive, lower is more similar

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two tensors."""
    dot_product = np.dot(a.flatten(), b.flatten())
    a_norm = np.linalg.norm(a.flatten())
    b_norm = np.linalg.norm(b.flatten())
    if a_norm == 0 and b_norm == 0:
        return 0.0
    elif a_norm == 0 or b_norm == 0:
        return 1.0
    else:
        return 1.0 - ((dot_product / (a_norm * b_norm)) + 1.0) / 2.0

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L1 distance between two tensors."""
    return np.sum(np.abs(a.flatten() - b.flatten()))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two tensors."""
    return np.linalg.norm(a.flatten() - b.flatten())


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation-based distance between two tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    correlation = np.corrcoef(a_flat, b_flat)[0, 1]
    if np.isnan(correlation):
        return 1.0
    return 1.0 - (correlation + 1.0) / 2.0


# Aggregation metrics inspired by the reference implementation
def mean_aggregation(scores: np.ndarray) -> float:
    """Mean aggregation of scores."""
    return np.mean(scores, axis=0)


def min_aggregation(scores: np.ndarray) -> float:
    """Minimum aggregation of scores."""
    return np.min(scores, axis=0)


def max_aggregation(scores: np.ndarray) -> float:
    """Maximum aggregation of scores."""
    return np.max(scores, axis=0)


def median_aggregation(scores: np.ndarray) -> float:
    """Median aggregation of scores."""
    return np.median(scores, axis=0)


# Metric constructors
def get_distance_metric(metric_type: str) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Get distance metric function by type.
    
    Args:
        metric_type (str): Type of distance metric ("cosine", "euclidean", "correlation")
        
    Returns:
        Callable: Distance metric function
        
    Raises:
        ValueError: If metric_type is not recognized
    """
    distance_metrics = {
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
        "correlation": correlation_distance,
    }
    
    if metric_type not in distance_metrics:
        raise ValueError(f"Unknown distance metric: {metric_type}. Available: {list(distance_metrics.keys())}")
    
    return distance_metrics[metric_type]


def get_aggregate_metric(metric_type: str) -> Callable[[np.ndarray], float]:
    """
    Get aggregation metric function by type.
    
    Args:
        metric_type (str): Type of aggregation metric ("mean", "min", "max", "median")
        
    Returns:
        Callable: Aggregation metric function
        
    Raises:
        ValueError: If metric_type is not recognized
    """
    aggregate_metrics = {
        "mean": mean_aggregation,
        "min": min_aggregation,
        "max": max_aggregation,
        "median": median_aggregation,
    }
    
    if metric_type not in aggregate_metrics:
        raise ValueError(f"Unknown aggregate metric: {metric_type}. Available: {list(aggregate_metrics.keys())}")
    
    return aggregate_metrics[metric_type]


def get_available_metrics() -> Dict[str, list]:
    """
    Get all available metric types.
    
    Returns:
        Dict[str, list]: Dictionary with available metric types for each category
    """
    return {
        "distance_metrics": ["cosine", "euclidean", "correlation"],
        "aggregate_metrics": ["mean", "min", "max", "median"],
    }


# Metric registry for easy access
class MetricRegistry:
    """Registry for all policy comparison metrics."""
    
    def __init__(self):
        self.distance_metrics = {
            "cosine": cosine_distance,
            "euclidean": euclidean_distance,
            "l1": l1_distance,
            "correlation": correlation_distance,
        }
        
        self.aggregate_metrics = {
            "mean": mean_aggregation,
            "min": min_aggregation,
            "max": max_aggregation,
            "median": median_aggregation,
        }
    
    def get_distance_metric(self, metric_type: str) -> Callable[[torch.Tensor, torch.Tensor], float]:
        """Get distance metric function."""
        if not set(metric_type).issubset(self.distance_metrics):
            raise ValueError(f"Unknown distance metric: {metric_type}. Available: {list(self.distance_metrics.keys())}")
        return [self.distance_metrics[mt] for mt in metric_type]
    
    def get_aggregate_metric(self, metric_type: str) -> Callable[[np.ndarray], float]:
        """Get aggregation metric function."""
        if metric_type not in self.aggregate_metrics:
            raise ValueError(f"Unknown aggregate metric: {metric_type}. Available: {list(self.aggregate_metrics.keys())}")
        return self.aggregate_metrics[metric_type]
    
    def list_available_metrics(self) -> Dict[str, list]:
        """List all available metric types."""
        return {
            "distance_metrics": list(self.distance_metrics.keys()),
            "aggregate_metrics": list(self.aggregate_metrics.keys()),
        }


# Global registry instance
metric_registry = MetricRegistry() 
import time
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np

class SystemEvaluator:
    """Tracks and aggregates execution times for Data Prep and Routing."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "data_preparation": [],
            "routing": [],
            "delay_annotation": [],
            "total_planning": []
        }

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time a specific block using high-resolution timer."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(elapsed)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all tracked operations."""
        summary = {}
        for op, values in self.metrics.items():
            if not values:
                continue
            summary[op] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)) if len(values) >= 20 else float(np.max(values)),
                "total": float(np.sum(values))
            }
        return summary

    def reset(self):
        """Clear all metrics."""
        for key in self.metrics:
            self.metrics[key] = []

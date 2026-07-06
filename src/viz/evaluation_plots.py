import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

def plot_calibration_curve(results: Dict[float, float], title: str = "Empirical Calibration Curve"):
    """Plot Promised Confidence vs Actual Success Rate.
    
    Args:
        results: Dictionary mapping {promised_q: actual_success_rate}
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    qs = sorted(results.keys())
    success_rates = [results[q] for q in qs]
    
    plt.plot(qs, success_rates, marker='o', linestyle='-', linewidth=2, label="Actual Success Rate")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    
    plt.xlabel("Promised Confidence (Q)")
    plt.ylabel("Actual Success Rate")
    plt.title(title)
    plt.xlim(0.4, 1.0)
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pareto_frontier(pareto_data: pd.DataFrame, title: str = "The Paranoia Tax (Pareto Frontier)"):
    """Plot Extra Travel Time vs Reliability.
    
    Args:
        pareto_data: DataFrame with columns ['success_rate', 'extra_time_min']
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    sns.lineplot(data=pareto_data, x='extra_time_min', y='success_rate', marker='o')
    
    plt.xlabel("Extra Travel Time (Minutes)")
    plt.ylabel("Probability of Success")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_latency_breakdown(latency_stats: Dict[str, float]):
    """Plot a bar chart of system performance components.
    
    Args:
        latency_stats: Dict with keys like 'Data Preparation', 'Routing Scan', 'Delay Annotation'
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    labels = list(latency_stats.keys())
    values = list(latency_stats.values())
    
    colors = sns.color_palette("viridis", len(labels))
    plt.barh(labels, values, color=colors)
    
    plt.xlabel("Execution Time (Seconds)")
    plt.title("System Performance Breakdown")
    plt.tight_layout()
    plt.show()

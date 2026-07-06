import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate prediction bias (mean error)."""
    return float(np.mean(y_pred - y_true))

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculate Pinball Loss (Quantile Loss) for a given quantile.
    
    Formula: L = (1 - alpha) * (y_pred - y_true) if y_pred > y_true else alpha * (y_true - y_pred)
    """
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

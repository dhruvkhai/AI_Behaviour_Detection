import numpy as np
from collections import Counter

class PostProcessor:
    """
    Utilities for refining model predictions using temporal context.
    """
    
    @staticmethod
    def majority_voting(predictions: list, window_size: int = 5) -> list:
        """
        Applies majority voting over a sliding window to smooth predictions.
        """
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start:end]
            most_common = Counter(window).most_common(1)[0][0]
            smoothed.append(most_common)
        return smoothed

    @staticmethod
    def temporal_smoothing(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Applies Exponential Moving Average (EMA) to smooth anomaly scores.
        formula: s[t] = alpha * x[t] + (1 - alpha) * s[t-1]
        """
        smoothed = np.zeros_like(scores)
        smoothed[0] = scores[0]
        for t in range(1, len(scores)):
            smoothed[t] = alpha * scores[t] + (1 - alpha) * smoothed[t-1]
        return smoothed

    @staticmethod
    def threshold_logic(scores: np.ndarray, threshold: float = -0.5) -> np.ndarray:
        """
        Converts anomaly scores to binary flags based on a threshold.
        """
        return (scores < threshold).astype(int)

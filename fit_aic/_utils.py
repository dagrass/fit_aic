import numpy as np


def _compute_aic_aicc(rss: float, n: int, k: int) -> tuple[float, float]:
    """Compute AIC and AICc for a given RSS, number of observations, and number of parameters."""
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("inf")
    return aic, aicc

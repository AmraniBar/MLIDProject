"""
utils_dynamic.py

Purpose:
    Provide helper functions for the dynamic extension, including:
     - KL divergence between Bernoulli distributions
     - Time-varying fluctuation functions (if needed)

Usage:
    from utils_dynamic import kl_divergence_bernoulli, ...
"""

import numpy as np

def kl_divergence_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    """
    Compute KL divergence between two Bernoulli(p) and Bernoulli(q) distributions.

    Args:
        p (float): parameter p in [0,1]
        q (float): parameter q in [0,1]
        eps (float): small value to avoid log(0)

    Returns:
        float: KL divergence KL(p||q)
    """
    if p < eps:
        if q < eps:
            return 0.0
        return (1 - p) * np.log((1 - p) / (1 - q + eps))
    elif p > 1 - eps:
        if q > 1 - eps:
            return 0.0
        return p * np.log((p) / (q + eps))
    else:
        return p * np.log(p/(q+eps)) + (1 - p)*np.log((1-p)/((1-q)+eps))

def fluctuation_function_sin(t: int, period: float = 50.0) -> float:
    """
    Example fluctuation function that returns a sinusoidal value in [-1, 1].
    You can scale it externally or use it directly.

    Args:
        t (int): current time step
        period (float): period for sinusoidal wave

    Returns:
        float: sin(2πt / period)
    """
    return np.sin((2.0 * np.pi * t) / period)

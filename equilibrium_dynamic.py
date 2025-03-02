"""
equilibrium_dynamic.py

Purpose:
    Computes the instantaneous Nash equilibrium (m*) for the dynamic scenario,
    given the current array of mu[].

Explanation:
    We reuse the standard binary search approach used in static MPMAB,
    but recast it for the time-varying mu. We do not store mu[] over time
    here; we just compute the equilibrium from the input mu at that moment.
"""

import numpy as np

def compute_nash_equilibrium_dynamic(mu, N, tol=1e-5, max_iter=1000):
    """
    Binary search to find z* so sum_{k=1..K} floor(mu_k / z*) >= N.
    Then m_k = floor(mu_k / z*). This is repeated each time we want to see
    the new equilibrium at the current mu.

    Args:
        mu (array-like): shape (K,). Current arms' expected values.
        N (int): total number of players.
        tol (float): tolerance for stopping the binary search.
        max_iter (int): maximum iterations for binary search.

    Returns:
        z_star (float), m_star (list of int):
            z_star: the critical threshold
            m_star: the integer allocation to each arm
    """
    mu = np.array(mu, dtype=float)
    lo, hi = 0.0, max(mu)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if mid < 1e-12:
            mid = 1e-12

        # m_k = floor(mu_k / mid)
        allocation = np.floor(mu / mid)
        total_alloc = np.sum(allocation)

        if total_alloc >= N:
            lo = mid
        else:
            hi = mid

        if abs(hi - lo) < tol:
            break

    z_star = lo
    m_star = [int(np.floor(x / z_star)) for x in mu]
    return z_star, m_star

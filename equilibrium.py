import numpy as np


def compute_nash_equilibrium(mu, N, tol=1e-5, max_iter=1000):
    """
    Compute the Nash equilibrium for the multi-player multi-armed bandit problem.

    According to the paper, the Nash equilibrium is characterized by an allocation m*
    where for each arm k:
        m*_k = floor(mu_k / z*)
    and z* is defined as:
        z* = sup { z > 0 : sum_{k=1}^{K} floor(mu_k / z) >= N }.

    This function uses binary search to approximate z*.

    Parameters:
        mu (list or np.ndarray): Array of expected rewards for each arm.
        N (int): Total number of players.
        tol (float): Tolerance for convergence in the binary search.
        max_iter (int): Maximum number of iterations for binary search.

    Returns:
        z_star (float): The critical value z*.
        m_star (list): The equilibrium allocation for each arm (list of integers)
    """
    # Convert mu to a NumPy array if not already
    mu = np.array(mu, dtype=float)
    # Set lower bound (lo) and upper bound (hi) for z
    lo, hi = 0.0, max(mu)

    # Perform binary search to approximate z*
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        # Avoid division by zero; if mid is extremely small, set allocation to a high number
        allocation = np.floor(mu / mid) if mid > 0 else np.full(mu.shape, np.inf)
        total_allocation = np.sum(allocation)

        if total_allocation >= N:
            lo = mid  # We can try a higher z value.
        else:
            hi = mid  # z is too high, lower it.

        # Check for convergence of the binary search
        if abs(hi - lo) < tol:
            break

    z_star = lo
    # Compute equilibrium allocation m* for each arm
    m_star = [int(np.floor(m / z_star)) for m in mu]

    return z_star, m_star


# # ------------------------------
# # Testing the Equilibrium Module
# # ------------------------------
# if __name__ == "__main__":
#     test_cases = [
#         {"mu": [1.0, 0.4, 0.2], "N": 3, "description": "Example 3.1 from paper (3 arms, 3 players)"},
#         {"mu": [0.9, 0.5, 0.3, 0.2], "N": 5, "description": "4 arms with varied rewards, 5 players"},
#         {"mu": [0.7, 0.7, 0.7, 0.7], "N": 4, "description": "Uniform rewards, 4 arms, 4 players"},
#         {"mu": [1, 1, 1, 1, 1], "N": 10, "description": "Identical rewards, 5 arms, 10 players"},
#     ]
#
#     for case in test_cases:
#         mu_test = case["mu"]
#         N_test = case["N"]
#         description = case["description"]
#         z_star, m_star = compute_nash_equilibrium(mu_test, N_test)
#         total_alloc = sum(m_star)
#         print(f"\nTest Case: {description}")
#         print("Expected rewards (mu):", mu_test)
#         print("Number of players (N):", N_test)
#         print("Computed z*:", z_star)
#         print("Equilibrium allocation m*:", m_star)
#         print("Total allocation:", total_alloc, "(should be >= N)")

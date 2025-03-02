import numpy as np

def kl_divergence(p, q):
    """
    Compute the KL divergence between two Bernoulli distributions with parameters p and q.
    Returns:
        KL(p||q)
    """
    # Handling edge cases:
    if p == 0:
        return (1 - p) * np.log((1 - p) / (1 - q))
    if p == 1:
        return p * np.log(p / q)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_instant_regret(arm_choices, env_mu, num_players):
    """
    Compute the instantaneous regret for each player at this round.
    Returns a 1D array of length num_players with each player's regret for the current round.
    """
    K = len(env_mu)
    counts = np.zeros(K, dtype=int)
    for a in arm_choices:
        counts[a] += 1

    regrets = np.zeros(num_players)
    for j, chosen_arm in enumerate(arm_choices):
        # Actual reward if staying on chosen arm:
        r_actual = env_mu[chosen_arm] / counts[chosen_arm] if counts[chosen_arm] > 0 else 0.0

        # Best alternative reward:
        best_alt = 0.0
        for k in range(K):
            if k == chosen_arm:
                alt = env_mu[k] / counts[k] if counts[k] > 0 else 0.0
            else:
                alt = env_mu[k] / (counts[k] + 1)
            if alt > best_alt:
                best_alt = alt

        regrets[j] = best_alt - r_actual
    return regrets


def is_equilibrium(arm_choices, env_mu, num_players):
    """
    Check if this round's arm choices match the Nash equilibrium counts
    computed from env_mu. Return True if it is equilibrium, else False.
    """
    from equilibrium import compute_nash_equilibrium

    K = len(env_mu)
    _, m_star = compute_nash_equilibrium(env_mu, num_players)
    # Count actual distribution
    counts = np.zeros(K, dtype=int)
    for a in arm_choices:
        counts[a] += 1

    # Compare counts with m_star
    return np.array_equal(counts, np.array(m_star))
import numpy as np
from equilibrium import compute_nash_equilibrium
import math


def compute_regret(global_arm_choices, global_rewards, env_mu, num_players):
    """
    Compute the cumulative regret for each player over T rounds.

    For each round t, let:
      - arm_choices[t] be the list of arms chosen by all players.
      - M_k(t) be the count of players choosing arm k.

    For player j at round t:
      - If player j chose arm k, his expected reward is:
          r_j = mu[k] / M_k(t)
      - If he deviates to an alternative arm k, his expected reward would be:
          r'_j = mu[k] / (M_k(t) + 1)   (because his deviation adds one extra pull)
      - His instantaneous regret is:
          reg_j(t) = max_{k in [K]} { mu[k] / (M_k(t) + delta(k, j) ) } - r_j,
        where delta(k, j) = 0 if k == chosen arm, else 1.

    Parameters:
        global_arm_choices (list of lists): Each inner list is the arm choices of all players at round t.
        global_rewards (list of lists): Each inner list is the rewards received by players at round t.
        env_mu (list or np.ndarray): True expected rewards for each arm.
        num_players (int): Total number of players.

    Returns:
        regrets (list): A list of cumulative regrets per player.
    """
    T = len(global_arm_choices)
    K = len(env_mu)
    regrets = np.zeros(num_players)

    for t in range(T):
        choices = global_arm_choices[t]
        rewards = global_rewards[t]
        # Compute counts: M_k(t) for each arm k.
        counts = np.zeros(K, dtype=int)
        for a in choices:
            counts[a] += 1

        for j in range(num_players):
            chosen_arm = choices[j]
            # Expected reward if staying on the chosen arm.
            if counts[chosen_arm] > 0:
                r_actual = env_mu[chosen_arm] / counts[chosen_arm]
            else:
                r_actual = 0.0

            # Compute candidate expected rewards for all arms.
            candidate_rewards = []
            for k in range(K):
                if k == chosen_arm:
                    candidate = env_mu[k] / counts[k] if counts[k] > 0 else 0.0
                else:
                    candidate = env_mu[k] / (counts[k] + 1)
                candidate_rewards.append(candidate)

            best_candidate = max(candidate_rewards)
            regrets[j] += best_candidate - r_actual

    return regrets.tolist()


def compute_non_equilibrium_rounds(global_arm_choices, env_mu, num_players):
    """
    Compute the number of non-equilibrium rounds.

    For each round t, let M(t) be the count vector of arms chosen by all players.
    Compute the Nash equilibrium allocation m* from the true expected rewards env_mu.
    If M(t) != m*, count that round as non-equilibrium.

    Parameters:
        global_arm_choices (list of lists): Each inner list is the arm choices of all players at round t.
        env_mu (list or np.ndarray): True expected rewards for each arm.
        num_players (int): Total number of players.

    Returns:
        non_equilibrium (int): Number of rounds that are not in equilibrium.
    """
    T = len(global_arm_choices)
    K = len(env_mu)
    # Compute Nash equilibrium allocation m* using true expected rewards.
    _, m_star = compute_nash_equilibrium(env_mu, num_players)

    non_equilibrium = 0
    for t in range(T):
        choices = global_arm_choices[t]
        counts = np.zeros(K, dtype=int)
        for a in choices:
            counts[a] += 1

        # Check if counts exactly match m_star.
        if not np.array_equal(counts, np.array(m_star)):
            non_equilibrium += 1

    return non_equilibrium

"""
metrics_dynamic.py

Purpose:
    Compute performance metrics under the dynamic model, including:
    - cumulative regret
    - non-equilibrium rounds
    - possibly custom metrics for time-varying scenarios
"""

import numpy as np
from equilibrium_dynamic import compute_nash_equilibrium_dynamic

def compute_dynamic_regret(global_arm_choices, global_personal_rewards, dynamic_mus, Phi, num_players):
    """
    Compute the cumulative regret for each player under time-varying, heterogeneous rewards.

    For each round t:
      - The environment has dynamic_mus[t]: shape (K,) for the base reward.
      - Phi is shape (N, K). So the effective mu for player j, arm k is dynamic_mus[t][k]*Phi[j,k].
      - If player j chose arm chosen_arm, their actual reward is global_personal_rewards[t][j].
      - We compare to the best alternative for that player at time t, which is:
          best_alt_j = max_k { dynamic_mus[t][k] * Phi[j,k] / (#players_on_arm_k(t) + 1 if k != chosen_arm
                                                               else #players_on_arm_k(t)) }

    Args:
        global_arm_choices (list of lists): arm choices at each round
        global_personal_rewards (list of lists): personal rewards at each round
        dynamic_mus (list or np.ndarray): dynamic base reward for each round, shape (T, K)
        Phi (np.ndarray): preference multipliers of shape (N, K)
        num_players (int): total players

    Returns:
        regrets (list): cumulative regret per player
    """
    T = len(global_arm_choices)
    K = len(dynamic_mus[0])
    regrets = np.zeros(num_players, dtype=float)

    for t in range(T):
        choices = global_arm_choices[t]
        personal_rewards_t = global_personal_rewards[t]
        base_mu_t = dynamic_mus[t]  # shape (K,)

        # Count how many players chose each arm
        counts = np.zeros(K, dtype=int)
        for a in choices:
            counts[a] += 1

        for j in range(num_players):
            chosen_arm = choices[j]
            r_actual = personal_rewards_t[j]

            # find best alternative
            best_alt = 0.0
            for k in range(K):
                # if k == chosen_arm, #players = counts[k]
                # else #players = counts[k] + 1
                if k == chosen_arm:
                    denom = counts[k] if counts[k] > 0 else 1
                else:
                    denom = (counts[k] + 1)
                eff_mu_jk = base_mu_t[k] * Phi[j, k] / denom
                if eff_mu_jk > best_alt:
                    best_alt = eff_mu_jk

            regrets[j] += (best_alt - r_actual)

    return regrets.tolist()

def compute_dynamic_non_equilibrium_rounds(global_arm_choices, dynamic_mus, num_players, tol=1e-8):
    """
    Compute the number of non-equilibrium rounds in a dynamic environment.
    For each round t:
      - We compute the NE from dynamic_mus[t].
      - If the chosen distribution of arms does not match that NE exactly,
        we increment the non-equilibrium count.

    This is a simplification since dynamic NE might shift frequently,
    but we keep the same definition as the static model to gauge
    how many times players exactly match the equilibrium counts.

    Args:
        global_arm_choices (list of lists): each inner list is arm choices of all players
        dynamic_mus (list of np.ndarray): shape (T, K) base reward at each round
        num_players (int): total players
        tol (float): tolerance in binary search for equilibrium

    Returns:
        non_equil (int): count of rounds that do not match the NE allocation
    """
    T = len(global_arm_choices)
    K = len(dynamic_mus[0])

    non_equil = 0

    for t in range(T):
        choices = global_arm_choices[t]
        base_mu_t = dynamic_mus[t]  # shape(K,)
        _, m_star = compute_nash_equilibrium_dynamic(base_mu_t, num_players, tol=tol)

        # count actual distribution
        counts = np.zeros(K, dtype=int)
        for arm in choices:
            counts[arm] += 1

        # compare to m_star
        if not np.array_equal(counts, np.array(m_star)):
            non_equil += 1

    return non_equil

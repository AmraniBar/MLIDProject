"""
simulation_dynamic.py

Purpose:
    Demonstrates how to configure and run a dynamic environment with multiple
    algorithms (including the updated SMAA and baselines). Then records metrics
    such as regrets and non-equilibrium rounds, and plots final results.
"""

import numpy as np
from environment_dynamic import DynamicMPMABEnvironment
from player_dynamic import (
    SMAAPlayerDynamic,
    ExploreThenCommitDynamic,
    TotalRewardDynamic,
    SelfishRobustMMABDynamic
)
from metrics_dynamic import (
    compute_dynamic_regret,
    compute_dynamic_non_equilibrium_rounds
)
from plot_results_dynamic import plot_final_bar_charts

def run_dynamic_simulation(env: DynamicMPMABEnvironment, players, T: int):
    """
    Runs the environment for T rounds. Each round:
      1) If t < K => initialization calls (player.initialize(t)).
      2) Else => block_data + choose_arm(t).
      3) environment.pull(...) => obtains personal rewards.
      4) players update(...) with personal reward.

    Returns:
      global_arm_choices, global_personal_rewards, dynamic_mus
        (lists capturing the entire run)
    """
    num_players = len(players)
    global_arm_choices = []
    global_personal_rewards = []
    dynamic_mus = []

    for t in range(T):
        # gather arms
        if t < env.K:
            arm_choices = [p.initialize(t) for p in players]
        else:
            block_data = []
            for p in players:
                # Some players have block_estimations or none
                if hasattr(p, 'block_estimations'):
                    block_data.append(p.block_estimations())
                else:
                    block_data.append(None)

            arm_choices = []
            for i, p in enumerate(players):
                try:
                    chosen_arm = p.choose_arm(t, block_data[i])
                except TypeError:
                    chosen_arm = p.choose_arm(t)
                arm_choices.append(chosen_arm)

        # environment step
        total_rewards, personal_rewards = env.pull(arm_choices, t)

        # update players
        for j, p in enumerate(players):
            p.update(arm_choices[j], personal_rewards[j])

        # record
        global_arm_choices.append(arm_choices)
        global_personal_rewards.append(personal_rewards)
        dynamic_mus.append(env.mu.copy())

    return global_arm_choices, global_personal_rewards, np.array(dynamic_mus)


def run_dynamic_experiment(
    K: int,
    N: int,
    T: int,
    dist: str,
    seed: int,
    player_classes: dict,
    delta_mu: float = 0.01
):
    """
    Orchestrates the creation of a dynamic environment and the chosen players,
    runs them for T rounds, and computes final metrics.

    Returns a dictionary with final regrets, non-equilibrium, etc. for each algorithm.
    """
    results = {}

    for alg_name, (alg_class, alg_params) in player_classes.items():
        # 1) Create environment
        env = DynamicMPMABEnvironment(
            K=K, N=N, T=T, dist=dist, seed=seed, delta_mu=delta_mu
        )

        # 2) Create players
        players = []
        for i in range(N):
            p = alg_class(
                player_id=i,
                K=K,
                T=T,
                N=N,
                seed=(seed + i*100),
                **alg_params
            )
            players.append(p)

        # 3) Run
        global_arm_choices, global_personal_rewards, dynamic_mus = run_dynamic_simulation(env, players, T)

        # 4) Evaluate final metrics
        regrets = compute_dynamic_regret(
            global_arm_choices, global_personal_rewards,
            dynamic_mus, env.Phi, N
        )
        ne_count = compute_dynamic_non_equilibrium_rounds(global_arm_choices, dynamic_mus, N)

        results[alg_name] = {
            'regrets': regrets,
            'non_equilibrium': ne_count,
            # Possibly track sum of EMA or final total
            'total_rewards': [np.sum(p.ema_means) for p in players]
        }

    return results

if __name__ == "__main__":
    # Example usage
    K = 10
    N = 8
    T = 10000
    dist = "beta"
    seed = 42
    delta_mu = 0.01

    player_classes = {
        "SMAA": (SMAAPlayerDynamic, {"beta": 0.1, "exploration_prob": 0.5, "ema_alpha": 0.1}),
        "ExploreThenCommit": (ExploreThenCommitDynamic, {"alpha": 10.0, "ema_alpha": 0.1}),
        "TotalReward": (TotalRewardDynamic, {"alpha": 10.0, "ema_alpha": 0.1}),
        "SelfishRobustMMAB": (SelfishRobustMMABDynamic, {"tau": 0.05, "ema_alpha": 0.1})
    }

    results = run_dynamic_experiment(
        K=K, N=N, T=T, dist=dist, seed=seed,
        player_classes=player_classes, delta_mu=delta_mu
    )

    # Generate bar charts
    plot_final_bar_charts(results)

    # Print final stats
    for alg, data in results.items():
        avg_reg = np.mean(data['regrets'])
        ne_rounds = data['non_equilibrium']
        print(f"\nAlgorithm: {alg}")
        print(f"  Avg Regret: {avg_reg:.2f}")
        print(f"  Non-Equilibrium Rounds: {ne_rounds}")
        print(f"  Total Rewards (ema_means sum): {np.mean(data['total_rewards']):.2f}")

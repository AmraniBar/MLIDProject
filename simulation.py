import math
import numpy as np
from tqdm import tqdm

from environment import MPMABEnvironment
from utils import is_equilibrium,compute_instant_regret
from player import SMAAPlayer, ExploreThenCommitPlayer,TotalRewardPlayer, SelfishRobustMMABPlayer
from plot_results import plot_time_series_scenario,plot_experiment_results

def run_simulation_time_series(player_class, num_players, K, T, dist='beta', seed=0, **kwargs):
    env = MPMABEnvironment(K=K, T=T, dist=dist, seed=seed)
    players = [
        player_class(player_id=i, K=K, T=T, N=num_players, seed=seed + i, **kwargs)
        for i in range(num_players)
    ]

    global_arm_choices = []
    global_shared_rewards = []   # if you want to track personal fraction
    partial_regrets = np.zeros((T, num_players))
    partial_non_eq = np.zeros(T, dtype=int)
    running_non_eq_count = 0
    desc_str = f"{player_class.__name__}-TimeSeries (dist={dist}, N={num_players}, K={K}, T={T})"
    for t in tqdm(range(T), desc=desc_str):
        # 1) Gather chosen arms
        if t < K:
            arm_choices = [p.initialize(t) for p in players]
        else:
            arm_choices = []
            for p in players:
                arm_choices.append(p.choose_arm(t))

        global_arm_choices.append(arm_choices)

        # 2) environment pulls => now returns (total_rewards, personal_rewards)
        total_rewards, personal_rewards = env.pull(arm_choices, t)
        global_shared_rewards.append(personal_rewards)

        # 3) Update each player using the *total* reward
        for i, p in enumerate(players):
            chosen_arm = arm_choices[i]
            p.update(chosen_arm, total_rewards[i])
            # ^ This is the crucial fix: pass total_rewards instead of personal_rewards

        # 4) Compute instant regret
        inst_regrets = compute_instant_regret(arm_choices, env.mu, num_players)
        if t == 0:
            partial_regrets[t] = inst_regrets
        else:
            partial_regrets[t] = partial_regrets[t-1] + inst_regrets

        # 5) Check equilibrium
        if not is_equilibrium(arm_choices, env.mu, num_players):
            running_non_eq_count += 1
        partial_non_eq[t] = running_non_eq_count

    # Return data
    return (global_arm_choices, global_shared_rewards, players, env,
            partial_regrets, partial_non_eq)



def paper_experiment():
    """
    Runs the full multi-seed experiment as in the paper.
    - Tracks cumulative regret and non-equilibrium rounds.
    - Runs multiple seeds per setting.
    - Generates & saves time-series plots + final bar charts.
    """

    # Experiment parameters
    N_values = [8, 20]  # Number of players
    K = 10  # Number of arms
    T = 500000  # Number of rounds
    num_seeds = 5  # Number of random seeds for robustness
    distributions = ["beta", "bernoulli"]  # Reward distributions
    seed_base = 42  # Base seed

    # Define algorithms to compare
    algorithms = {
        "SMAA": (SMAAPlayer, {"beta": 0.1, "exploration_prob": 0.5}),
        "ExploreThenCommit": (ExploreThenCommitPlayer, {"alpha": 10.0}),
        "TotalReward": (TotalRewardPlayer, {"alpha": 10.0}),
        "SelfishRobustMMAB": (SelfishRobustMMABPlayer, {"tau": 0.01}),
    }

    # Storage for aggregated results
    results = {}

    for N in N_values:
        for dist in distributions:
            results[(N, dist)] = {}

            for alg_name in algorithms:
                results[(N, dist)][alg_name] = {
                    "regret_list": [],
                    "non_eq_list": []
                }

            print(f"\n=== Running for N={N}, dist={dist} ===")

            for seed_id in range(num_seeds):
                current_seed = seed_base + 1000 * seed_id

                # Create environment with fixed seed
                env = MPMABEnvironment(K=K, T=T, dist=dist, seed=current_seed)

                for alg_name, (alg_class, alg_params) in algorithms.items():
                    # Reset environment for fair comparisons
                    env_local = MPMABEnvironment(K=K, T=T, dist=dist, seed=current_seed)

                    # Create players
                    players = [
                        alg_class(player_id=i, K=K, T=T, N=N, seed=current_seed + i, **alg_params)
                        for i in range(N)
                    ]

                    # Tracking arrays
                    partial_regrets = np.zeros((T, N), dtype=float)
                    partial_non_eq = np.zeros(T, dtype=int)
                    running_non_eq_count = 0

                    for t in tqdm(range(T), desc=f"{alg_name} (seed {seed_id})"):
                        arm_choices = [p.initialize(t) if t < K else p.choose_arm(t) for p in players]
                        total_rewards, personal_rewards = env_local.pull(arm_choices, t)

                        # Update players
                        for i, p in enumerate(players):
                            p.update(arm_choices[i], total_rewards[i])

                        # Compute cumulative regret
                        inst_regrets = compute_instant_regret(arm_choices, env_local.mu, N)
                        partial_regrets[t] = inst_regrets if t == 0 else partial_regrets[t-1] + inst_regrets

                        # Track non-equilibrium rounds
                        if not is_equilibrium(arm_choices, env_local.mu, N):
                            running_non_eq_count += 1
                        partial_non_eq[t] = running_non_eq_count

                    # Store results for this seed
                    final_regret = np.mean(partial_regrets[-1])  # Final regret per player
                    final_non_eq = partial_non_eq[-1]  # Final non-equilibrium count

                    results[(N, dist)][alg_name]["regret_list"].append(final_regret)
                    results[(N, dist)][alg_name]["non_eq_list"].append(final_non_eq)

    # Generate and save plots
    plot_experiment_results(results, N_values, K, T, distributions, num_seeds)

def test_smaa_experiment():
    """
    Run a small-scale time-series experiment.
    Plots partial regrets & partial non-equilibrium over time for each algorithm.
    """
    # Define your small scenario
    N = 8
    K = 10
    T = 500000
    dist = "beta"  # or "bernoulli"

    # # Define algorithms to compare
    algorithms = {
        "SMAA": (SMAAPlayer, {"beta": 0.1, "exploration_prob": 0.5}),
        "ExploreThenCommit": (ExploreThenCommitPlayer, {"alpha": 3.0}),
        "TotalReward": (TotalRewardPlayer, {"alpha": 10.0}),
        "SelfishRobustMMAB": (SelfishRobustMMABPlayer, {"tau": 0.01}),
    }
    # Dictionaries to store partial regrets & partial non-equilibrium
    partial_regrets_dict = {}
    partial_non_eq_dict = {}

    # For reproducibility
    seed = 42

    for alg_name, (alg_class, alg_params) in algorithms.items():
        print(f"\n=== Testing {alg_name} on (N={N}, K={K}, T={T}, dist={dist}) ===")
        # Use the run_simulation_time_series function to collect partial metrics
        (gac, grw, players, env,
         partial_regrets, partial_non_eq) = run_simulation_time_series(
             player_class=alg_class,
             num_players=N,
             K=K,
             T=T,
             dist=dist,
             seed=seed,
             **alg_params
        )

        # Store the partial regrets / partial non_eq for plotting
        partial_regrets_dict[alg_name] = partial_regrets
        partial_non_eq_dict[alg_name] = partial_non_eq

        # Print final cumulative regret & final non-equilibrium
        final_regrets = partial_regrets[-1]  # shape: (num_players,)
        final_non_eq = partial_non_eq[-1]
        print(f"{alg_name}: Final regrets per player = {final_regrets}, "
              f"Non-equilibrium rounds = {final_non_eq}")

    # Now plot the partial metrics over time
    # This will create two .png files: one for regret, one for non-equilibrium.
    plot_time_series_scenario(partial_regrets_dict, partial_non_eq_dict, N, K, T, dist)


if __name__ == "__main__":

    test_smaa_experiment()

    # paper_experiment()
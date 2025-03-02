import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_regrets(results):
    """
    Plot the final cumulative regrets per player for each algorithm as a bar chart.
    """
    algorithms = list(results.keys())
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        regrets = results[alg]['regrets']
        avg_regret = np.mean(regrets)
        plt.bar(alg, avg_regret, label=f"{alg} (avg: {avg_regret:.2f})")
    plt.title("Average Cumulative Regret per Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Cumulative Regret")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_total_rewards(results):
    """
    Plot the final total rewards per player for each algorithm as a bar chart.
    """
    algorithms = list(results.keys())
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        total_rewards = results[alg]['total_rewards']
        avg_reward = np.mean(total_rewards)
        plt.bar(alg, avg_reward, label=f"{alg} (avg: {avg_reward:.2f})")
    plt.title("Average Total Cumulative Reward per Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_non_equilibrium(results):
    """
    Plot the final total non-equilibrium rounds for each algorithm as a bar chart.
    """
    algorithms = list(results.keys())
    non_eq = [results[alg]['non_equilibrium'] for alg in algorithms]
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, non_eq, color='orange')
    plt.title("Total Non-Equilibrium Rounds per Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Non-Equilibrium Rounds")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_time_series_scenario(partial_regrets_dict, partial_non_eq_dict,
                              N_, K_, T_, dist_):
    """
    Create line charts for partial (cumulative) regrets and partial (cumulative) non-equilibrium
    over time, replicating the style of Figures 1 & 2 in the paper.

    partial_regrets_dict: { alg_name -> 2D np.array of shape (T, num_players) }
    partial_non_eq_dict:   { alg_name -> 1D np.array of shape (T,) }
    """
    # Plot cumulative regret over time (averaged across players)
    plt.figure(figsize=(6, 4))
    for alg_name, regrets_2d in partial_regrets_dict.items():
        avg_regrets = np.mean(regrets_2d, axis=1)  # shape (T,)
        plt.plot(avg_regrets, label=alg_name)
    plt.title(f"Regret over time (N={N_}, K={K_}, dist={dist_})")
    plt.xlabel("Round")
    plt.ylabel("Avg. Cumulative Regret")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname_reg = f"regret_N{N_}_K{K_}_{dist_}.png"
    plt.savefig(fname_reg)
    plt.close()

    # Plot cumulative non-equilibrium rounds over time
    plt.figure(figsize=(6, 4))
    for alg_name, non_eq_1d in partial_non_eq_dict.items():
        plt.plot(non_eq_1d, label=alg_name)
    plt.title(f"Non-equilibrium over time (N={N_}, K={K_}, dist={dist_})")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Non-Equilibrium Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname_ne = f"non_equ_N{N_}_K{K_}_{dist_}.png"
    plt.savefig(fname_ne)
    plt.close()



def plot_experiment_results(results, N_values, K, T, distributions, num_seeds):
    """
    Generates & saves the final bar charts and time-series plots.
    """

    for (N, dist), alg_results in results.items():
        plt.figure(figsize=(6, 4))

        # Compute average regret & non-equilibrium across seeds
        avg_regret_per_alg = {alg: np.mean(alg_results[alg]["regret_list"]) for alg in alg_results}
        std_regret_per_alg = {alg: np.std(alg_results[alg]["regret_list"]) for alg in alg_results}
        avg_noneq_per_alg = {alg: np.mean(alg_results[alg]["non_eq_list"]) for alg in alg_results}
        std_noneq_per_alg = {alg: np.std(alg_results[alg]["non_eq_list"]) for alg in alg_results}

        # Print summary statistics
        print(f"\nResults for N={N}, dist={dist}:")
        for alg in alg_results:
            print(f"  {alg}: Regret {avg_regret_per_alg[alg]:.2f} ± {std_regret_per_alg[alg]:.2f}, "
                  f"Non-Equilibrium {avg_noneq_per_alg[alg]:.1f} ± {std_noneq_per_alg[alg]:.1f}")

        # Plot final regret bar chart
        plt.bar(avg_regret_per_alg.keys(), avg_regret_per_alg.values(),
                yerr=std_regret_per_alg.values(), capsize=5)
        plt.title(f"Avg. Cumulative Regret (N={N}, dist={dist}, {num_seeds} seeds)")
        plt.ylabel("Regret")
        plt.xlabel("Algorithm")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"final_regret_N{N}_{dist}.png")
        plt.close()

        # Plot final non-equilibrium bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(avg_noneq_per_alg.keys(), avg_noneq_per_alg.values(),
                yerr=std_noneq_per_alg.values(), capsize=5, color="orange")
        plt.title(f"Total Non-Equilibrium Rounds (N={N}, dist={dist}, {num_seeds} seeds)")
        plt.ylabel("Non-Equilibrium Rounds")
        plt.xlabel("Algorithm")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"final_noneq_N{N}_{dist}.png")
        plt.close()

    print("\n✅ All plots saved successfully!")
# if __name__ == "__main__":
#     # This file only has plotting code. Typically, you'd run simulation_old.py
#     # and import these functions from there.
#     print("plot_results.py loaded. Use the provided functions for plotting.")

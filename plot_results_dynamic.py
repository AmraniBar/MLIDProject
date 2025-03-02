"""
plot_results_dynamic.py

Purpose:
    Functions to visualize final bar charts (average regret, non-equilibrium) for the
    dynamic scenario, as well as optional time-series plots if partial metrics are recorded.
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def plot_final_bar_charts(results):
    """
    Create bar charts for final average regrets and total non-equilibrium rounds,
    saving them to .png files.

    Args:
        results (dict): e.g. {
            alg_name: {
               'regrets': [...],
               'non_equilibrium': int,
               'total_rewards': [...],
            }, ...
        }
    """
    algorithms = list(results.keys())

    # 1) Regret chart
    avg_reg_list = [np.mean(results[alg]['regrets']) for alg in algorithms]
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, avg_reg_list, color='skyblue')
    for i, v in enumerate(avg_reg_list):
        plt.text(i, v+0.01, f"{v:.2f}", ha='center')
    plt.title("Average Cumulative Regret (Dynamic)")
    plt.xlabel("Algorithm")
    plt.ylabel("Regret")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dynamic_avg_regret.png")
    plt.close()

    # 2) Non-equilibrium chart
    ne_list = [results[alg]['non_equilibrium'] for alg in algorithms]
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, ne_list, color='orange')
    for i, v in enumerate(ne_list):
        plt.text(i, v+0.1, str(v), ha='center')
    plt.title("Total Non-Equilibrium Rounds (Dynamic)")
    plt.xlabel("Algorithm")
    plt.ylabel("Non-Equilibrium Rounds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dynamic_non_equilibrium.png")
    plt.close()

def plot_time_series_scenario(partial_regrets_dict, partial_non_eq_dict, N_, K_, T_, dist_):
    """
    If we record partial regrets and partial non-equilibrium each round, we can plot them.
    partial_regrets_dict: {alg_name: 2D array (T, N) or 1D array (T,) of cumulative regrets}
    partial_non_eq_dict:  {alg_name: 1D array (T,) of cumulative non-equilibrium counts}
    """
    # Plot partial regrets over time.
    plt.figure(figsize=(6,4))
    for alg_name, regrets_2d in partial_regrets_dict.items():
        if regrets_2d.ndim == 2:
            avg_regrets = np.mean(regrets_2d, axis=1)
        else:
            avg_regrets = regrets_2d
        plt.plot(avg_regrets, label=alg_name)

    plt.title(f"Regret over time (N={N_}, K={K_}, dist={dist_})")
    plt.xlabel("Round")
    plt.ylabel("Avg. Cumulative Regret")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"dynamic_regret_time_series_N{N_}_K{K_}_{dist_}.png")
    plt.close()

    # Plot partial non-equilibrium counts.
    plt.figure(figsize=(6,4))
    for alg_name, noneq_array in partial_non_eq_dict.items():
        plt.plot(noneq_array, label=alg_name)

    plt.title(f"Non-Equilibrium over time (N={N_}, K={K_}, dist={dist_})")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Non-Equilibrium Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"dynamic_non_eq_time_series_N{N_}_K{K_}_{dist_}.png")
    plt.close()

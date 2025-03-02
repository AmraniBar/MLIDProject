# ML-Decision Project: Dynamic Extension

## Overview
This repository implements the implement the methods presented in the paper " Competing for Shareable Arms in Multi-Player Multi-
Armed Bandits", and dynamic extension of the multi-player multi-armed bandit (MPMAB) framework. 
In the dynamic setting, the environment supports:
- **Time-varying rewards:** Each arm’s expected reward updates over time (e.g., via a sinusoidal fluctuation).
- **Heterogeneous player preferences:** Each player receives a personalized reward based on a preference matrix, so that even when sharing the same arm, rewards are scaled differently.

The project provides dynamic versions of all core components:
- **Environment:** Simulates a dynamic MPMAB setting with drifting arm parameters.
- **Equilibrium Computation:** Computes an instantaneous Nash equilibrium based on the current reward parameters.
- **Players:** Implements dynamic variants of the SMAA algorithm and several baseline strategies (Explore-Then-Commit, Total Reward, and SelfishRobustMMAB).
- **Metrics:** Computes cumulative regret and non-equilibrium rounds under dynamic conditions.
- **Visualization:** Generates plots (bar charts and time-series) for the recorded performance metrics.
- **Utilities:** Provides helper functions for dynamic updates (e.g., KL divergence for Bernoulli distributions, fluctuation functions).

## Project stracture:
.
├── environment_dynamic.py       # Dynamic MPMAB environment with time-varying rewards and player preferences
├── equilibrium_dynamic.py       # Computes instantaneous Nash equilibrium for the dynamic scenario
├── metrics_dynamic.py           # Performance metrics (regret and non-equilibrium rounds) for dynamic simulations
├── player_dynamic.py            # Dynamic player implementations (SMAADynamic, ExploreThenCommitDynamic, etc.)
├── plot_results_dynamic.py      # Functions to generate plots for dynamic simulation results
├── simulation_dynamic.py        # Main simulation routine to run dynamic experiments
├── utils_dynamic.py             # Utility functions for the dynamic extension (e.g., KL divergence, fluctuation functions)
└── README.md                    # This README file

## Requirements
- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)


## How to Run the Simulation ?
To run the dynamic simulation and generate results:
- Open a terminal in the repository directory.
- Run the main simulation script:
python simulation_dynamic.py

You can install the required packages using pip:
```bash
pip install numpy matplotlib tqdm


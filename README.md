# ML-Decision Project: Dynamic Extension

## Overview
This repository implements the methods presented in the paper "Competing for Shareable Arms in Multi-Player Multi-Armed Bandits" and dynamic extension of the multi-player multi-armed bandit (MPMAB) framework.

In the dynamic setting, the environment supports:
- **Time-varying rewards:** Each arm's expected reward updates over time (e.g., via a sinusoidal fluctuation)
- **Heterogeneous player preferences:** Each player receives a personalized reward based on a preference matrix, so that even when sharing the same arm, rewards are scaled differently

The project provides dynamic versions of all core components:
- **Environment:** Simulates a dynamic MPMAB setting with drifting arm parameters
- **Equilibrium Computation:** Computes an instantaneous Nash equilibrium based on the current reward parameters
- **Players:** Implements dynamic variants of the SMAA algorithm and several baseline strategies (Explore-Then-Commit, Total Reward, and SelfishRobustMMAB)
- **Metrics:** Computes cumulative regret and non-equilibrium rounds under dynamic conditions
- **Visualization:** Generates plots (bar charts and time-series) for the recorded performance metrics
- **Utilities:** Provides helper functions for dynamic updates (e.g., KL divergence for Bernoulli distributions, fluctuation functions)

## Project Structure
```
├── environment_dynamic.py       # Dynamic MPMAB environment with time-varying rewards and player preferences
├── equilibrium_dynamic.py       # Computes instantaneous Nash equilibrium for the dynamic scenario
├── metrics_dynamic.py           # Performance metrics (regret and non-equilibrium rounds) for dynamic simulations
├── player_dynamic.py            # Dynamic player implementations (SMAADynamic, ExploreThenCommitDynamic, etc.)
├── plot_results_dynamic.py      # Functions to generate plots for dynamic simulation results
├── simulation_dynamic.py        # Main simulation routine to run dynamic experiments
├── utils_dynamic.py             # Utility functions for the dynamic extension (e.g., KL divergence, fluctuation functions)
└── README.md                    # This README file
```

## Requirements
- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Installation
You can install the required packages using pip:
```bash
pip install numpy matplotlib tqdm
```

## How to Run the Simulation
To run the dynamic simulation and generate results:
1. Clone this repository:
```bash
git clone https://github.com/yourusername/ml-decision-dynamic.git
cd ml-decision-dynamic
```

2. Run the main simulation script:
```bash
python simulation_dynamic.py
```

## Key Features
- **Dynamic Reward Modeling**: Time-dependent reward distributions that change according to configurable patterns
- **Preference-based Rewards**: Player-specific reward scaling to model heterogeneous agent preferences
- **Adaptive Algorithms**: Enhanced learning algorithms that can adapt to changing environments
- **Comprehensive Metrics**: Performance evaluation under dynamic conditions with time-series analysis

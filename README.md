
# Multi-Player Multi-Armed Bandits (MPMAB) – SMAA Implementation

This repository implements the Selfish Multi-Player Multi-Armed Bandit (MPMAB) algorithm with Averaging Allocation (SMAA) as described in the paper *"Competing for Shareable Arms in Multi-Player Multi-Armed Bandits"*. The project includes various player strategies, environment simulations, equilibrium computation, and experimental validation of asymptotic performance bounds by tracking cumulative regret and non-equilibrium rounds.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Implementation File Descriptions](#Implementation_File_Descriptions)
  - [mlds_project_report.pdf](#mlds_project_reportpdf)
  - [environment.py](#environmentpy)
  - [equilibrium.py](#equilibriumpy)
  - [utils.py](#utilspy)
  - [metrics.py](#metricspy)
  - [player.py](#playerpy)
  - [simulation.py](#simulationpy)
  - [plot_results.py](#plot_resultspy)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [Test SMAA Experiment](#test-smaa-experiment)
  - [Full-Scale Experiment](#full-scale-experiment)
- [Requirements](#requirements)
- [ ML-Decision Project: Dynamic Extension](#ML-Decision_Project:_Dynamic_Extension)
- [License](#license)

## Overview

In multi-player multi-armed bandit problems, several self-interested players compete for arms whose rewards are shareable. When multiple players choose the same arm, the reward is divided among them. Our implementation uses the SMAA algorithm that:

- **Estimates Arm Quality:** Each player computes an empirical mean for every arm.
- **Computes Nash Equilibrium:** A binary search procedure finds the critical threshold (z*) and equilibrium allocation (m*), ensuring the total allocation covers the number of players.
- **Balances Exploration and Exploitation:** A block-based strategy with KL-UCB exploration helps refine decisions over time.
- **Tracks Performance:** Both cumulative regret and non-equilibrium rounds (i.e., rounds where the arm selection deviates from the Nash equilibrium) are monitored to demonstrate asymptotic performance bounds.

The accompanying project report (`mlds_project_report.pdf`) provides a detailed theoretical foundation, experimental analysis, and discussion of potential extensions.

## Repository Structure


## Project Structure
```
├── mlds_project_report.pdf       # Detailed project report with theoretical background and experimental results
├── environment.py                # Simulation of the multi-player bandit environment with shareable rewards
├── equilibrium.py                # Computation of Nash equilibrium (z* and m* allocation)
├── utils.py                      # Utility functions (KL divergence, instant regret, equilibrium check)
├── metrics.py                    # Functions to compute cumulative regret and non-equilibrium rounds
├── player.py                     # Implementation of SMAA and baseline player strategies
├── simulation.py                 # Code to run experiments and simulations (time series and full-scale)
├── plot_results.py               # Plotting utilities for visualization of simulation results
└── requirements.txt              # List of Python package dependencies
├── environment_dynamic.py       # Dynamic MPMAB environment with time-varying rewards and player preferences
├── equilibrium_dynamic.py       # Computes instantaneous Nash equilibrium for the dynamic scenario
├── metrics_dynamic.py           # Performance metrics (regret and non-equilibrium rounds) for dynamic simulations
├── player_dynamic.py            # Dynamic player implementations (SMAADynamic, ExploreThenCommitDynamic, etc.)
├── plot_results_dynamic.py      # Functions to generate plots for dynamic simulation results
├── simulation_dynamic.py        # Main simulation routine to run dynamic experiments
├── utils_dynamic.py             # Utility functions for the dynamic extension (e.g., KL divergence, fluctuation functions)
└── README.md                    # This README file
```


## Implementation File Descriptions

### mlds_project_report.pdf
This file contains the complete project report that covers:
- The theoretical model, assumptions, and equilibrium analysis.
- A detailed description of the SMAA algorithm.
- The experimental setup, results, and performance metrics.
- A discussion of possible extensions (e.g., dynamic rewards and heterogeneous player preferences).

### environment.py
Defines the `MPMABEnvironment` class which:
- Simulates a bandit environment where each arm’s reward is drawn from either a Beta or Bernoulli distribution.
- Shares the sampled reward equally among all players that choose the same arm.
- Also includes an extended environment (e.g., `PeriodicPreferenceEnvironment`) for dynamic or periodic reward functions.

### equilibrium.py
Provides the function `compute_nash_equilibrium` that:
- Uses a binary search to determine the threshold value (z*) such that each arm’s allocation is given by  
  `m*_k = floor(μ_k / z*)`.
- Ensures that the sum of the allocations meets or exceeds the number of players.
- Outputs both the computed z* and the equilibrium allocation vector (m*).

### utils.py
Contains helper routines:
- **`kl_divergence(p, q)`**: Computes the KL divergence between two Bernoulli distributions.
- **`compute_instant_regret(arm_choices, env_mu, num_players)`**: Calculates the instantaneous regret for each player based on the difference between the obtained reward and the best alternative reward.
- **`is_equilibrium(arm_choices, env_mu, num_players)`**: Checks if the current arm choices match the Nash equilibrium distribution computed from the expected rewards.

### metrics.py
Implements functions to evaluate performance:
- **`compute_regret(...)`**: Computes the cumulative regret per player across all rounds.
- **`compute_non_equilibrium_rounds(...)`**: Counts the number of rounds in which the actual arm distribution deviates from the Nash equilibrium allocation.

### player.py
Implements various player strategies:
- **SMAAPlayer:**  
  - The core implementation of the SMAA algorithm.
  - Uses a block-based approach to periodically compute the equilibrium allocation and determine candidate arms for exploration based on KL-divergence thresholds.
- **SMAAMusicalChairsPlayer:**  
  - A variant that employs a "musical chairs" strategy to handle situations where the number of players (N) is unknown or partially known.
- **ExploreThenCommitPlayer and TotalRewardPlayer:**  
  - Baseline strategies that first explore and then commit to the best-performing arm based on empirical rewards.
- Additional variants (e.g., SelfishRobustMMABPlayer) are also available for comparative evaluation.

### simulation.py
Contains the code for running experiments:
- **`run_simulation_time_series`**:  
  - Simulates a bandit game over T rounds.
  - Records each round’s arm choices, cumulative (partial) regret, and non-equilibrium rounds.
- **`test_smaa_experiment`**:  
  - A small-scale experiment that demonstrates asymptotic performance by tracking partial regrets and non-equilibrium rounds over time.
- **`paper_experiment`**:  
  - Executes a full-scale experiment (as described in the paper) using multiple seeds and varying parameters.
  - **Note:** For a quick test, use `test_smaa_experiment()`. To run the full-scale experiment, adjust the main function accordingly.

### plot_results.py
Provides plotting functions to visualize results:
- Generates time-series plots of cumulative regret and non-equilibrium rounds.
- Produces bar charts summarizing final performance metrics for different algorithms.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo```
**install the Required Packages:**
```bash
pip install -r requirements.txt

```
##Running Experiments
##Test SMAA Experiment
This experiment runs a small-scale simulation that demonstrates:

Logarithmic cumulative regret growth.
Rapid convergence to the Nash equilibrium (indicated by a drop in non-equilibrium rounds).
Steps:
## Running Experiments

Edit the Main Block in `simulation.py`:
Open `simulation.py` and ensure that the main function calls `test_smaa_experiment()` rather than `paper_experiment()`. For example:

```python
if __name__ == "__main__":
    # To run the full-scale experiment, you might call:
    # paper_experiment()

    # For testing the SMAA experiment:
    test_smaa_experiment()
```
##Run the Simulation:
```bash
python simulation.py

```

##full-Scale Experiment
The full-scale experiment replicates the experimental setup described in the paper, featuring:

Multiple random seeds for robust statistical evaluation.
Tracking of both cumulative regret and non-equilibrium rounds.
Comparison across several algorithms (SMAA, ExploreThenCommit, TotalReward, etc.).
Steps:

Modify the Main Block in simulation.py:
Comment out the call to test_smaa_experiment() and uncomment the call to paper_experiment():

```python
if __name__ == "__main__":
    # Run the full-scale experiment:
    paper_experiment()

```

**Run the Simulation:**
```bash
python simulation.py

```


## ML-Decision Project: Dynamic Extension


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

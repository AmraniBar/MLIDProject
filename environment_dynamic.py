"""
environment_dynamic.py

Purpose:
    Defines a multi-player multi-armed bandit environment that supports:
      1) Time-varying rewards for each arm (Beta or Bernoulli).
      2) Heterogeneous player preferences (matrix Phi).

Core Changes from the Original SMAA:
    - The function _update_arm_parameters(t) modifies each arm's mu_k(t) each round,
      e.g., using a sine wave drift or any user-defined function.
    - Preference matrix Phi ensures that different players get different
      scaled rewards from the same base sample.
"""

import numpy as np

class DynamicMPMABEnvironment:
    """
    Environment with K arms and N players, over T total rounds:
      - dist: 'beta' or 'bernoulli' to define initial distribution.
      - mu[] is updated each round to model drift in reward quality.
      - Phi (N x K) is a preference matrix, so each player's reward is:
            (base_sample / collision_count) * Phi[j, arm].
    """

    def __init__(
        self,
        K: int,
        N: int,
        T: int,
        dist: str = 'beta',
        seed: int = 42,
        delta_mu: float = 0.01,
        preference_seed: int = 123,
    ):
        """
        Args:
            K (int): Number of arms.
            N (int): Number of players.
            T (int): Total rounds (for scheduling, if needed).
            dist (str): 'beta' or 'bernoulli' distribution for arms.
            seed (int): random seed for arm initialization and updates.
            delta_mu (float): magnitude of drift in time-varying updates.
            preference_seed (int): random seed used to generate preference matrix Phi.
        """
        self.K = K
        self.N = N
        self.T = T
        self.dist = dist
        self.rng = np.random.default_rng(seed)
        self.delta_mu = delta_mu

        # A separate RNG for preference matrix, for clarity.
        self.pref_rng = np.random.default_rng(preference_seed)

        # Initialize arms (alpha/beta for Beta, or mu for Bernoulli).
        self._initialize_arms()

        # Create the preference matrix Phi of shape (N,K).
        self._initialize_preferences()

    def _initialize_arms(self):
        """Set up initial parameters for arms, depending on 'beta' or 'bernoulli'."""
        if self.dist == 'beta':
            self.alpha = self.rng.uniform(0.5, 5.0, self.K)
            self.beta = self.rng.uniform(0.5, 5.0, self.K)
            self.mu = self.alpha / (self.alpha + self.beta)
        elif self.dist == 'bernoulli':
            self.mu = self.rng.uniform(0, 1, self.K)
        else:
            raise ValueError("dist must be 'beta' or 'bernoulli'.")

    def _initialize_preferences(self):
        """Generate a preference matrix Phi of shape (N,K). Each entry is in [0.5,1.5] by default."""
        self.Phi = 0.5 + self.pref_rng.random((self.N, self.K))

    def _update_arm_parameters(self, t: int):
        """
        Update mu[] each round to simulate time-varying rewards.

        Here, we add a sinusoidal shift of amplitude delta_mu * sin(2πt/50).
        Beta arms: shift alpha (or directly shift mu). Bernoulli arms: shift mu in [0,1].
        """
        fluctuation = np.sin(2 * np.pi * t / 50.0)
        if self.dist == 'beta':
            shift = self.delta_mu * fluctuation
            self.alpha += shift
            # Keep alpha > 0.1 for numerical stability
            self.alpha = np.clip(self.alpha, 0.1, 9999.0)
            self.mu = self.alpha / (self.alpha + self.beta)
        else:  # 'bernoulli'
            shift = self.delta_mu * fluctuation
            self.mu += shift
            self.mu = np.clip(self.mu, 0.0, 1.0)

    def pull(self, arm_choices, t: int):
        """
        Called each round. The environment:
          1) Updates mu via _update_arm_parameters(t).
          2) Counts collisions.
          3) Each player's base reward is sampled. It's shared if multiple players pick the same arm.
          4) Multiply by Phi[j,arm] to get personal reward.

        Args:
            arm_choices (list of int): chosen arms by each player j.
            t (int): current round index.

        Returns:
            total_rewards (np.ndarray): shape (N,), each player's base reward before preference
            personal_rewards (np.ndarray): shape (N,), each player's final reward after preference
        """
        self._update_arm_parameters(t)

        counts = np.zeros(self.K, dtype=int)
        for arm in arm_choices:
            counts[arm] += 1

        total_rewards = np.zeros(self.N, dtype=float)
        personal_rewards = np.zeros(self.N, dtype=float)

        for j, arm in enumerate(arm_choices):
            # sample base reward from the chosen arm
            if self.dist == 'beta':
                base_sample = self.rng.beta(self.alpha[arm], self.beta[arm])
            else:
                base_prob = self.mu[arm]
                base_sample = float(self.rng.binomial(1, base_prob))

            # share among collisions
            share = base_sample / counts[arm] if counts[arm] > 0 else 0.0
            total_rewards[j] = share
            # multiply by preference
            personal_rewards[j] = share * self.Phi[j, arm]

        return total_rewards, personal_rewards

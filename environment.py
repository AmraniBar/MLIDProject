import numpy as np


# class MPMABEnvironment:
#     """
#     Multi-Player Multi-Armed Bandit (MPMAB) Environment.
#
#     This class simulates a bandit environment where each arm k has a reward
#     distribution (Beta or Bernoulli) with expected value mu_k. When multiple
#     players pull the same arm at a round, the generated reward is shared equally
#     among them.
#     """
#
#     def __init__(self, K, T, dist='beta', seed=None):
#         """
#         Initialize the MPMAB environment.
#
#         Parameters:
#             K (int): Number of arms.
#             T (int): Total number of rounds (not used directly in reward generation,
#                      but can be useful for scheduling experiments).
#             dist (str): Distribution type ('beta' or 'bernoulli').
#             seed (int or None): Random seed for reproducibility.
#         """
#         self.K = K
#         self.T = T
#         self.dist = dist
#         self.rng = np.random.default_rng(seed)
#         self._initialize_arms()
#
#     def _initialize_arms(self):
#         """
#         Initialize the arm parameters and compute expected rewards.
#
#         For Beta distribution:
#             - Randomly sample alpha and beta parameters from a uniform range.
#             - Compute mu_k = alpha_k / (alpha_k + beta_k).
#
#         For Bernoulli distribution:
#             - Randomly sample mu_k (the probability of success) from a uniform [0,1].
#         """
#         if self.dist == 'beta':
#             # Sample parameters for each arm; these ranges can be adjusted.
#             self.alpha = self.rng.uniform(0.5, 5.0, self.K)
#             self.beta = self.rng.uniform(0.5, 5.0, self.K)
#             self.mu = self.alpha / (self.alpha + self.beta)
#         elif self.dist == 'bernoulli':
#             self.mu = self.rng.uniform(0, 1, self.K)
#         else:
#             raise ValueError("Unsupported distribution type: choose 'beta' or 'bernoulli'.")
#
#     def pull(self, arm_choices, t):
#         """
#         Simulate a round of arm pulls.
#
#         Parameters:
#             arm_choices (list or array): A list of chosen arm indices, one per player.
#             t (int): Current round (optional for more dynamic models).
#
#         Returns:
#             rewards (list): A list of rewards for each player. If multiple players choose
#                             the same arm, they share the arm's reward equally.
#         """
#         # Count the number of players choosing each arm.
#         counts = np.zeros(self.K, dtype=int)
#         for a in arm_choices:
#             counts[a] += 1
#
#         rewards = []
#         for a in arm_choices:
#             # Draw a reward sample from the selected arm's distribution.
#             if self.dist == 'beta':
#                 reward = self.rng.beta(self.alpha[a], self.beta[a])
#             else:  # Bernoulli distribution
#                 reward = self.rng.binomial(1, self.mu[a])
#
#             # Share the reward equally if multiple players chose the same arm.
#             reward_shared = reward / counts[a] if counts[a] > 0 else 0
#             rewards.append(reward_shared)
#         return rewards
import numpy as np

class MPMABEnvironment:
    """
    Multi-Player Multi-Armed Bandit (MPMAB) Environment.

    This class simulates a bandit environment where each arm k has a reward
    distribution (Beta or Bernoulli) with expected value mu_k. When multiple
    players pull the same arm at a round, the generated reward is shared equally
    among them.
    """

    def __init__(self, K, T, dist='beta', seed=None):
        """
        Initialize the MPMAB environment.

        Parameters:
            K (int): Number of arms.
            T (int): Total number of rounds.
            dist (str): Distribution type ('beta' or 'bernoulli').
            seed (int or None): Random seed for reproducibility.
        """
        self.K = K
        self.T = T
        self.dist = dist
        self.rng = np.random.default_rng(seed)
        self._initialize_arms()

    def _initialize_arms(self):
        """
        Initialize the arm parameters and compute expected rewards.

        For Beta distribution:
            - Randomly sample alpha and beta parameters from a uniform range.
            - Compute mu_k = alpha_k / (alpha_k + beta_k).

        For Bernoulli distribution:
            - Randomly sample mu_k (the probability of success) from a uniform [0,1].
        """
        if self.dist == 'beta':
            # Sample parameters for each arm; these ranges can be adjusted.
            self.alpha = self.rng.uniform(0.5, 5.0, self.K)
            self.beta = self.rng.uniform(0.5, 5.0, self.K)
            self.mu = self.alpha / (self.alpha + self.beta)
        elif self.dist == 'bernoulli':
            self.mu = self.rng.uniform(0, 1, self.K)
        else:
            raise ValueError("Unsupported distribution type: choose 'beta' or 'bernoulli'.")

    def pull(self, arm_choices, t):
        """
        Simulate a round of arm pulls.

        Parameters:
            arm_choices (list or array): A list of chosen arm indices, one per player.
            t (int): Current round (optional, can be used for dynamic models).

        Returns:
            total_rewards (np.ndarray): total_rewards[i] is the entire sampled reward
                                        from the arm chosen by player i.
            personal_rewards (np.ndarray): personal_rewards[i] is the share of the
                                           arm's reward allocated to player i, based
                                           on how many players chose that arm.
        """
        num_players = len(arm_choices)
        counts = np.zeros(self.K, dtype=int)
        for arm in arm_choices:
            counts[arm] += 1

        total_rewards = np.zeros(num_players)
        personal_rewards = np.zeros(num_players)

        # For each player, sample once from the chosen arm
        for i, arm in enumerate(arm_choices):
            if self.dist == 'beta':
                # Full sample from Beta(alpha[arm], beta[arm])
                sample_reward = self.rng.beta(self.alpha[arm], self.beta[arm])
            else:
                # Bernoulli with prob mu[arm]
                sample_reward = self.rng.binomial(1, self.mu[arm])

            total_rewards[i] = sample_reward
            # Share equally if multiple players chose the same arm
            if counts[arm] > 0:
                personal_rewards[i] = sample_reward / counts[arm]
            else:
                personal_rewards[i] = 0.0

        return total_rewards, personal_rewards



# # ------------------------------
# # Testing the Environment Module
# # ------------------------------
# if __name__ == "__main__":
#     # Test parameters
#     num_arms = 5
#     total_rounds = 10
#     seed = 123
#
#     # Create environment instance using Beta distribution
#     env_beta = MPMABEnvironment(K=num_arms, T=total_rounds, dist='beta', seed=seed)
#     print("Beta Distribution - Expected Rewards (mu):", env_beta.mu)
#
#     # Test case 1: Each player chooses a different arm (no collision)
#     arm_choices_no_collision = list(range(num_arms))
#     rewards_no_collision = env_beta.pull(arm_choices_no_collision, t=1)
#     print("\nTest Case 1 (No Collision):")
#     print("Arm choices:", arm_choices_no_collision)
#     print("Rewards:", rewards_no_collision)
#
#     # Test case 2: Multiple players choose the same arm (collision)
#     # For example, let 4 players choose arm 0, and 1 player choose arm 1.
#     arm_choices_collision = [0, 0, 0, 0, 1]
#     rewards_collision = env_beta.pull(arm_choices_collision, t=2)
#     print("\nTest Case 2 (Collision):")
#     print("Arm choices:", arm_choices_collision)
#     print("Rewards:", rewards_collision)
#
#     # Create environment instance using Bernoulli distribution
#     env_bern = MPMABEnvironment(K=num_arms, T=total_rounds, dist='bernoulli', seed=seed)
#     print("\nBernoulli Distribution - Expected Rewards (mu):", env_bern.mu)
#
#     # Run a similar test for Bernoulli
#     rewards_bern = env_bern.pull(arm_choices_collision, t=3)
#     print("\nBernoulli Test Case (Collision):")
#     print("Arm choices:", arm_choices_collision)
#     print("Rewards:", rewards_bern)

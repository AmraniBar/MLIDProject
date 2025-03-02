"""
player_dynamic.py

Purpose:
    Implements dynamic players. The main highlight is the SMAAPlayerDynamic,
    which uses an exponential moving average of personal rewards and block-based
    re-check of the equilibrium to handle time-varying mu and heterogeneous preferences.
"""

import math
import numpy as np
from equilibrium_dynamic import compute_nash_equilibrium_dynamic
from utils_dynamic import kl_divergence_bernoulli

class SMAAPlayerDynamic:
    """
    SMAA-based player for a time-varying, heterogeneous preference scenario.

    Key logic:
      - We keep an exponential moving average (ema_means) for each arm,
        representing the player's estimate of arm quality from their perspective.
      - block_size = N => we re-check equilibrium every N rounds, though you can
        adapt this to other intervals or thresholds if desired.
    """

    def __init__(
        self,
        player_id: int,
        K: int,
        T: int,
        N: int,
        seed: int = 42,
        beta: float = 0.1,
        exploration_prob: float = 0.5,
        ema_alpha: float = 0.1,
    ):
        """
        Args:
            player_id (int): unique ID of the player in [0..N-1].
            K (int): total arms.
            T (int): total rounds.
            N (int): total players.
            seed (int): random seed.
            beta (float): factor for potential KL-based exploration (unused in example).
            exploration_prob (float): probability of exploring if we're the designated explorer.
            ema_alpha (float): exponential moving average factor for personal reward updates.
        """
        self.player_id = player_id
        self.K = K
        self.T = T
        self.N = N
        self.beta = beta
        self.exploration_prob = exploration_prob
        self.ema_alpha = ema_alpha

        self.rng = np.random.default_rng(seed)

        self.ema_means = np.zeros(K, dtype=float)
        self.counts_arm = np.zeros(K, dtype=int)

        # For block-based logic:
        self.block_size = N
        self.current_block = -1
        self.block_arm_sequence = []
        self.r_hat_min_arm = 0
        self.explore_idx = -1
        self.candidate = []

        self.history = []
        self.coin_index = 0
        # Pre-generate random coins
        self.explore_coins = self.rng.random(self.T)

    def initialize(self, t: int) -> int:
        """During the first K steps, do a simple approach (round-robin or random)."""
        arm = t % self.K
        self.history.append(arm)
        return arm

    def update(self, chosen_arm: int, personal_reward: float):
        """
        Incorporate the new personal reward via an exponential moving average.
        This ensures we adapt to changes quickly if ema_alpha is relatively large.
        """
        self.counts_arm[chosen_arm] += 1
        old_val = self.ema_means[chosen_arm]
        self.ema_means[chosen_arm] = (1 - self.ema_alpha)*old_val + self.ema_alpha*personal_reward

    def block_estimations(self):
        """
        Called each round in some frameworks, but here we do nothing.
        Equilibrium re-check is done in choose_arm(...) if t//N changes.
        """
        pass

    def choose_arm(self, t: int, block_data=None) -> int:
        """
        Each round:
         - if it's a new block => re-compute eq
         - pick the block_arm_sequence for that index
         - if we're the designated explorer => we might pick from candidate
        """
        if t < self.K:
            return self.initialize(t)

        block_id = t // self.block_size
        if block_id != self.current_block:
            self.current_block = block_id
            self._compute_equilibrium_and_candidates(t)

        idx_in_block = (self.player_id + t) % self.N
        if len(self.block_arm_sequence) > 0:
            arm = self.block_arm_sequence[idx_in_block % len(self.block_arm_sequence)]
        else:
            # fallback if m_star is all zero
            arm = self.rng.integers(0, self.K)

        if idx_in_block == self.explore_idx:
            # Possibly explore
            if self.coin_index < len(self.explore_coins):
                if self.explore_coins[self.coin_index] < self.exploration_prob and len(self.candidate) > 0:
                    arm = self.rng.choice(self.candidate)
                self.coin_index += 1

        self.history.append(arm)
        return arm

    def _compute_equilibrium_and_candidates(self, t: int):
        """
        1) Using self.ema_means as mu[].
        2) compute m_star using equilibrium_dynamic
        3) build block_arm_sequence (N arms in total).
        4) find the min r_hat arm => designated explorer
        5) define candidate arms for exploration (any arm with m_star[k] = 0).
        """
        # 1) equilibrium
        z_star, m_star = compute_nash_equilibrium_dynamic(self.ema_means, self.N)

        # 2) build the arms for a block
        block_list = []
        for k in range(self.K):
            block_list.extend([k]*m_star[k])

        leftover = self.N - len(block_list)
        if leftover > 0:
            # fill leftover with best arms
            best_arms_desc = np.argsort(-self.ema_means)
            idx_fill = 0
            while leftover > 0:
                block_list.append(best_arms_desc[idx_fill % self.K])
                idx_fill += 1
                leftover -= 1

        self.block_arm_sequence = block_list

        # 3) candidate = arms not in eq but we've tried them
        self.candidate = []
        for k in range(self.K):
            if m_star[k] == 0 and self.counts_arm[k] > 0:
                # optionally check if kl_val <= threshold, but here we add all
                self.candidate.append(k)

        # 4) min-arm in eq => we designate that as the explorer
        eq_arms = [k for k in range(self.K) if m_star[k] > 0]
        if len(eq_arms) > 0:
            eq_arms_sorted = sorted(eq_arms, key=lambda x: self.ema_means[x])
            min_arm = eq_arms_sorted[0]
        else:
            min_arm = np.argmin(self.ema_means)

        # find the index positions in block_list
        positions = [i for i,a in enumerate(block_list) if a == min_arm]
        if positions:
            self.explore_idx = positions[0]
        else:
            self.explore_idx = 0


"""
Below are dynamic baseline players that rely on an exponential moving average
and re-check logic, or a collision penalty approach.

ExploreThenCommitDynamic
TotalRewardDynamic
SelfishRobustMMABDynamic
"""
import math
import numpy as np

class ExploreThenCommitDynamic:
    """
    Explore-Then-Commit for a dynamic environment:
      - For alpha*log(T) steps, do random exploration.
      - Then commit to best arm by personal average (exponential moving average).
      - Re-check occasionally if environment might have changed significantly.
    """

    def __init__(self, player_id, K, T, N, seed=42, alpha=10.0, recheck_interval=None, ema_alpha=0.1):
        self.player_id = player_id
        self.K = K
        self.T = T
        self.N = N
        self.rng = np.random.default_rng(seed)

        self.alpha = alpha
        self.T_explore = max(1, int(alpha * math.log(T))) if T>1 else 1
        self.ema_alpha = ema_alpha

        if recheck_interval is None:
            self.recheck_interval = max(1, 2*self.T_explore)
        else:
            self.recheck_interval = recheck_interval

        self.counts = np.zeros(K, dtype=int)
        self.ema_means = np.zeros(K, dtype=float)
        self.committed_arm = None
        self.history = []

    def initialize(self, t):
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t):
        if t < self.T_explore:
            # random exploration
            arm = self.rng.integers(0, self.K)
        else:
            if self.committed_arm is None:
                self.committed_arm = int(np.argmax(self.ema_means))

            # re-check
            if (t - self.T_explore) > 0 and (t - self.T_explore) % self.recheck_interval == 0:
                curr_val = self.ema_means[self.committed_arm]
                best_alt = np.max(self.ema_means)
                if best_alt > curr_val + 0.01:
                    self.committed_arm = int(np.argmax(self.ema_means))

            arm = self.committed_arm

        self.history.append(arm)
        return arm

    def update(self, chosen_arm, personal_reward):
        self.counts[chosen_arm] += 1
        old_val = self.ema_means[chosen_arm]
        self.ema_means[chosen_arm] = (1 - self.ema_alpha)*old_val + self.ema_alpha*personal_reward


class TotalRewardDynamic:
    """
    Similar to ExploreThenCommit but focuses on 'total reward' as a measure.
    Implementation is effectively the same as ExploreThenCommitDynamic
    but semantically can differ in how "committed_arm" is chosen or re-checked.
    """

    def __init__(self, player_id, K, T, N, seed=42, alpha=10.0, recheck_interval=None, ema_alpha=0.1):
        self.player_id = player_id
        self.K = K
        self.T = T
        self.N = N
        self.rng = np.random.default_rng(seed)

        self.alpha = alpha
        self.T_explore = max(1, int(alpha * math.log(T))) if T>1 else 1
        self.ema_alpha = ema_alpha

        if recheck_interval is None:
            self.recheck_interval = max(1, 2*self.T_explore)
        else:
            self.recheck_interval = recheck_interval

        self.counts = np.zeros(K, dtype=int)
        self.ema_means = np.zeros(K, dtype=float)
        self.committed_arm = None
        self.history = []

    def initialize(self, t):
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t):
        if t < self.T_explore:
            arm = self.rng.integers(0, self.K)
        else:
            if self.committed_arm is None:
                self.committed_arm = int(np.argmax(self.ema_means))

            # re-check
            if (t - self.T_explore) > 0 and (t - self.T_explore) % self.recheck_interval == 0:
                curr_val = self.ema_means[self.committed_arm]
                best_alt = np.max(self.ema_means)
                if best_alt > curr_val + 0.01:
                    self.committed_arm = int(np.argmax(self.ema_means))

            arm = self.committed_arm

        self.history.append(arm)
        return arm

    def update(self, chosen_arm, personal_reward):
        self.counts[chosen_arm] += 1
        old_val = self.ema_means[chosen_arm]
        self.ema_means[chosen_arm] = (1 - self.ema_alpha)*old_val + self.ema_alpha*personal_reward


class SelfishRobustMMABDynamic:
    """
    A collision-averse approach in a dynamic environment.
    - We maintain an exponential moving average of personal reward.
    - If personal_reward < collision_threshold * current_mean => we treat it as a collision event.
    - Then we apply a penalty factor in choose_arm by incrementing collision_counts.
    """

    def __init__(self, player_id, K, T, N, seed=42, tau=0.05, ema_alpha=0.1, collision_threshold=0.7):
        self.player_id = player_id
        self.K = K
        self.T = T
        self.N = N
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.collision_threshold = collision_threshold

        self.rng = np.random.default_rng(seed)

        self.ema_means = np.zeros(K, dtype=float)
        self.counts = np.zeros(K, dtype=int)
        self.collision_counts = np.zeros(K, dtype=int)
        self.history = []

    def initialize(self, t: int) -> int:
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t: int) -> int:
        """
        Weighted softmax over (ema_means[k] - penalty).
        Higher collision_counts => more penalty => less chance to pick that arm.
        """
        adjusted_values = []
        for k in range(self.K):
            penalty_fraction = self.collision_counts[k] / max(1, self.counts[k])
            # The more collisions, the more we reduce effective estimate
            adjusted_val = max(0.0, self.ema_means[k] - 0.5 * penalty_fraction)
            adjusted_values.append(adjusted_val)

        exps = np.exp(np.array(adjusted_values) / self.tau)
        probs = exps / np.sum(exps)

        arm = self.rng.choice(self.K, p=probs)
        self.history.append(arm)
        return arm

    def update(self, chosen_arm: int, personal_reward: float):
        old_val = self.ema_means[chosen_arm]
        self.ema_means[chosen_arm] = (1 - self.ema_alpha)*old_val + self.ema_alpha*personal_reward
        self.counts[chosen_arm] += 1

        # If personal_reward < threshold*current_mean => increment collision count
        if personal_reward < self.collision_threshold * self.ema_means[chosen_arm]:
            self.collision_counts[chosen_arm] += 1

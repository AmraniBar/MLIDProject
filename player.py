

import math
import numpy as np
from equilibrium import compute_nash_equilibrium
from utils import kl_divergence


################################################################
##  1. SMAAPlayer
################################################################
class SMAAPlayer:
    """
    SMAA (Selfish MPMAB with Averaging Allocation) Player.

    This class implements the main ideas from the paper:
    "Competing for Shareable Arms in Multi-Player Multi-Armed Bandits."
    Specifically, it models a selfish player who:
      - Estimates the reward of each arm (µ_k).
      - Computes the approximate Nash Equilibrium (m*) once every N rounds.
      - Follows a block-based approach to exploit arms in the estimated equilibrium,
        with occasional exploration based on KL divergence thresholds.

    Key Highlights:
    1. We accept total_reward in update(...), reflecting the fact that arms are shareable
       and each collision yields an expected fraction of the arm’s reward.
    2. We recalculate the estimated NE (m*) only once per block of size N.
    3. We use KL-based checks to explore arms that might still be viable candidates.
    """

    def __init__(self, player_id, K, T, N,
                 seed=0, beta=0.1, exploration_prob=0.5, tolerance=1e-6):
        """
        Initializes the SMAAPlayer with parameters for the MPMAB setting.

        Args:
            player_id (int): Unique ID of this player in [0..N-1].
            K (int): Number of arms.
            T (int): Total number of rounds (used for random coin flips).
            N (int): Total number of players (assumed known in this version).
            seed (int): Random seed for reproducibility.
            beta (float): Scaling factor for the KL threshold (used in exploration logic).
            exploration_prob (float): Probability of exploring if this player is the "explorer" in the block.
            tolerance (float): Used for borderline checks in the NE computation.

        Attributes:
            sum_rewards_arm (np.ndarray): Sum of total rewards obtained from each arm.
            counts_arm (np.ndarray): Number of times each arm has been chosen.
            history (list): Records the arms chosen at each round.
            rng (np.random.Generator): Random number generator for coin flips, exploration, etc.
            coin (np.ndarray): Pre-generated random floats for deciding exploration (length T).
            coin_index (int): Index into the coin array.
            KK (int): Number of rounds for short initialization phase (ensures each arm is sampled).
            pne_list (list): Holds the repeated arms from the computed equilibrium for the next block.
            candidate (list): Candidate arms to explore if they meet KL-based criteria.
            explore_idx (int): Which player in the block can explore the candidate set (if any).
        """
        self.player_id = player_id
        self.K = K
        self.T = T
        self.N = N
        self.beta = beta
        self.exploration_prob = exploration_prob
        self.tolerance = tolerance

        self.sum_rewards_arm = np.zeros(K)
        self.counts_arm = np.zeros(K, dtype=int)
        self.history = []

        self.rng = np.random.default_rng(seed)
        self.coin = self.rng.random(T)  # random floats in [0,1)
        self.coin_index = 0

        # Compute an integer multiple for the initialization block
        self.KK = int(np.ceil(K / N)) * N

        # For block-based logic
        self.pne_list = []
        self.candidate = []
        self.explore_idx = -1

    def initialize(self, t):
        """
        Called during early phases (e.g., first K rounds) to ensure each arm is sampled.

        Args:
            t (int): Current round index.

        Returns:
            int: The arm chosen at round t (simple round-robin).
        """
        arm = t % self.K
        self.history.append(arm)
        return arm

    def block_estimations(self):
        """
        Optional hook for logic that might be called every round to update block-level data.
        In this implementation, it's a no-op, since block computations occur in choose_arm(...).
        """
        pass

    def choose_arm(self, t, block_data=None):
        """
        Selects which arm to pull at round t, using the SMAA block-based strategy.

        Args:
            t (int): Current round index.
            block_data (Any): Additional block-level data (unused by default here).

        Returns:
            int: The index of the chosen arm.
        """
        if t < self.K:
            # Same as initialize phase
            arm = t % self.K
        elif t < self.KK:
            # Pure random exploration during the short initialization block
            arm = self.rng.integers(0, self.K)
        else:
            # Every new block of size N, compute the approximate NE
            if t % self.N == 0:
                self._compute_equilibrium_and_candidates(t)

            # Determine which arm in the equilibrium list to choose
            idx = (self.player_id + t) % self.N
            if idx < len(self.pne_list):
                arm = self.pne_list[idx]
            else:
                arm = self.rng.integers(0, self.K)  # Fallback if pne_list is empty

            # Possibly explore if we are the designated explorer and have candidate arms
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_index] < self.exploration_prob:
                    arm = self.rng.choice(self.candidate)
                self.coin_index += 1

        self.history.append(arm)
        return arm

    def _compute_equilibrium_and_candidates(self, t):
        """
        1) Estimate µ_k for each arm k.
        2) Compute NE (z*, m*) from equilibrium.py (the approximate Nash Equilibrium).
        3) Build a repeated list of arms in the NE (pne_list).
        4) Build a candidate set for exploration if the KL condition is satisfied.
        5) Decide which player index in [0..N-1] can do exploration.

        Args:
            t (int): The current round index (used for log factors in KL thresholds).
        """
        mu_est = np.zeros(self.K)
        for k in range(self.K):
            if self.counts_arm[k] > 0:
                mu_est[k] = self.sum_rewards_arm[k] / self.counts_arm[k]

        # Compute approximate NE
        z_star, m_star = compute_nash_equilibrium(mu_est, self.N)

        # Build repeated list of arms from m_star
        self.pne_list = []
        for k in range(self.K):
            for _ in range(m_star[k]):
                self.pne_list.append(k)

        # Identify candidate arms not in the NE but potentially worth exploring
        self.candidate = []
        for k in range(self.K):
            if m_star[k] == 0 and self.counts_arm[k] > 0:
                kl_val = self.counts_arm[k] * kl_divergence(mu_est[k], z_star)
                threshold = self.beta * (
                    math.log(t+1) + 4 * math.log(max(2, math.log(t+1)))
                )
                if kl_val <= threshold:
                    self.candidate.append(k)

        # Decide which player in [0..N-1] might do exploration
        self.explore_idx = -1
        if len(self.pne_list) == self.N:
            # The "worst" arm in the NE is the designated explorer
            # i.e., smallest µ among arms in the NE
            sorted_indices = sorted(range(self.N),
                                    key=lambda i: mu_est[self.pne_list[i]])
            self.explore_idx = sorted_indices[0]

    def update(self, chosen_arm, total_reward):
        """
        Updates internal statistics after each round with the total reward from the chosen arm.

        Args:
            chosen_arm (int): The index of the arm pulled.
            total_reward (float): The entire reward sampled from that arm (shared among players).
        """
        self.sum_rewards_arm[chosen_arm] += total_reward
        self.counts_arm[chosen_arm] += 1


################################################################
##  2. SMAAMusicalChairsPlayer
##     A variant that inherits from SMAAPlayer, but uses
##     "musical chairs" logic for unknown N or partial feedback.
################################################################
class SMAAMusicalChairsPlayer(SMAAPlayer):
    """
    A derived class from SMAAPlayer that incorporates a "musical chairs" style
    approach to handle cases where the number of players (N) might be unknown or
    partially known. This approach can also help break ties more robustly.

    Reference:
        See Section 3.4 of the paper, where the authors combine SMAA with a
        Musical Chairs method (Rosenski et al., 2016) to handle unknown N.
    """

    def __init__(self, player_id, K, T, N=None,
                 seed=0, beta=0.1, exploration_prob=0.5, tolerance=1e-6):
        """
        Initializes the musical-chairs variant of SMAA.

        Args:
            player_id (int): Unique ID of this player in [0..N-1] (or a guess if N is unknown).
            K (int): Number of arms.
            T (int): Total rounds.
            N (int or None): Total number of players (can be None if unknown).
            seed (int): Random seed for reproducibility.
            beta (float): Scaling factor for KL threshold.
            exploration_prob (float): Probability of exploring in block-based logic.
            tolerance (float): Tolerance used for borderline checks.

        Notes:
            If N is None, we might do an active approach to estimate it, or treat it as very large.
        """
        if N is None:
            # Placeholder for demonstration. In a real system, you'd implement logic to discover N.
            N = 9999999
        super().__init__(player_id, K, T, N,
                         seed=seed, beta=beta,
                         exploration_prob=exploration_prob,
                         tolerance=tolerance)
        self.mc_phase_done = False
        self.rank = player_id  # Could be used if we had a real rank assignment step

    def initialize(self, t):
        """
        In the musical-chairs phase, each player tries different arms to find a conflict-free slot.

        Args:
            t (int): Current round index.

        Returns:
            int: Chosen arm for initialization.
        """
        arm = super().initialize(t)
        return arm

    def choose_arm(self, t, block_data=None):
        """
        Overrides the choose_arm method to incorporate a short "musical chairs" phase,
        then falls back to the standard SMAA approach.

        Args:
            t (int): Current round index.
            block_data (Any): Additional block data (unused here).

        Returns:
            int: The index of the chosen arm.
        """
        if not self.mc_phase_done and t < self.K:
            # Simple approach: each player tries a different arm by offsetting player_id
            arm = (t + self.player_id) % self.K
            self.history.append(arm)
            return arm

        if t == self.K:
            # After K steps, assume the musical chairs phase is done
            self.mc_phase_done = True

        # Fallback to the parent SMAA logic
        return super().choose_arm(t, block_data)


##########################################################################
## ExploreThenCommitPlayer
##########################################################################
class ExploreThenCommitPlayer:
    """
    Explore-Then-Commit baseline:
      - Explores arms randomly for alpha * log(T) rounds,
      - Then commits to the arm with the highest empirical average reward.
      - Optionally re-checks occasionally to handle collisions.

    This approach is simpler than SMAA and does not explicitly compute a Nash equilibrium,
    making it less sophisticated but easier to implement.
    """

    def __init__(self, player_id, K, T, alpha=10.0, seed=None, N=None):
        """
        Args:
            player_id (int): Unique ID of the player.
            K (int): Number of arms.
            T (int): Total rounds.
            alpha (float): Controls the exploration length (~ alpha * log(T)).
            seed (int): Random seed for reproducibility.
            N (int): Optional number of players (unused by default here).
        """
        self.player_id = player_id
        self.K = K
        self.T = T
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K)
        self.history = []

        self.T_explore = int(alpha * math.log(T)) if T > 1 else 1
        self.committed_arm = None

        # For periodic re-check
        self.recheck_interval = max(1, 2 * self.T_explore)

    def initialize(self, t):
        """
        Called during early rounds to randomly explore arms.

        Args:
            t (int): Current round index.

        Returns:
            int: The chosen arm (random).
        """
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t):
        """
        After the exploration phase, commit to the best arm found so far,
        re-checking periodically in case collisions degrade performance.

        Args:
            t (int): Current round index.

        Returns:
            int: The chosen arm index.
        """
        if t < self.T_explore:
            # Still exploring
            arm = self.rng.integers(0, self.K)
        else:
            # Commit if not set
            if self.committed_arm is None:
                means = self._empirical_means()
                self.committed_arm = int(np.argmax(means))

            # Re-check occasionally
            if (t - self.T_explore) > 0 and ((t - self.T_explore) % self.recheck_interval == 0):
                current_mean = self._empirical_means()[self.committed_arm]
                means = self._empirical_means()
                best_alt = np.max(means)
                if best_alt > current_mean + 0.01:
                    self.committed_arm = int(np.argmax(means))

            arm = self.committed_arm

        self.history.append(arm)
        return arm

    def update(self, chosen_arm, total_reward):
        """
        Update empirical statistics for the chosen arm.

        Args:
            chosen_arm (int): The arm that was pulled this round.
            total_reward (float): The total reward sampled from that arm.
        """
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += total_reward

    def _empirical_means(self):
        """
        Computes empirical means for each arm.

        Returns:
            np.ndarray: An array of shape (K,) with the average reward of each arm.
        """
        return np.divide(self.sum_rewards, np.maximum(1, self.counts))


##########################################################################
## TotalRewardPlayer
##########################################################################
class TotalRewardPlayer:
    """
    A baseline approach similar to ExploreThenCommit, but specifically focuses
    on maximizing personal "total reward" and includes a small re-check mechanism
    to handle collisions or shifts in arm quality.

    It also explores for alpha * log(T) rounds, then commits to the best arm.
    """

    def __init__(self, player_id, K, T, alpha=10.0, seed=None, N=None):
        """
        Args:
            player_id (int): Unique ID of the player.
            K (int): Number of arms.
            T (int): Total rounds.
            alpha (float): Controls the exploration length (~ alpha * log(T)).
            seed (int): Random seed for reproducibility.
            N (int): Optional number of players.
        """
        self.player_id = player_id
        self.K = K
        self.T = T
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K)
        self.history = []

        self.T_explore = int(alpha * math.log(T)) if T > 1 else 1
        self.committed_arm = None

        # Periodic re-check
        self.recheck_interval = max(1, 2 * self.T_explore)

    def initialize(self, t):
        """
        Random exploration in the early rounds.

        Args:
            t (int): Current round index.
        Returns:
            int: The chosen arm.
        """
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t):
        """
        Similar to ExploreThenCommit, but we interpret 'total reward' to
        guide the final commitment. We also do a small re-check step.

        Args:
            t (int): Current round index.

        Returns:
            int: The chosen arm index.
        """
        if t < self.T_explore:
            arm = self.rng.integers(0, self.K)
        else:
            if self.committed_arm is None:
                means = self._empirical_means()
                self.committed_arm = int(np.argmax(means))

            if (t - self.T_explore) > 0 and ((t - self.T_explore) % self.recheck_interval == 0):
                current_mean = self._empirical_means()[self.committed_arm]
                means = self._empirical_means()
                best_alt = np.max(means)
                if best_alt > current_mean + 0.01:
                    self.committed_arm = int(np.argmax(means))

            arm = self.committed_arm

        self.history.append(arm)
        return arm

    def update(self, chosen_arm, total_reward):
        """
        Updates the sum of rewards and counts for the chosen arm.

        Args:
            chosen_arm (int): Index of the arm pulled.
            total_reward (float): The total reward from that arm (could be a share if multiple players chose it).
        """
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += total_reward

    def _empirical_means(self):
        """
        Computes empirical means for each arm.

        Returns:
            np.ndarray: Average reward array of shape (K,).
        """
        return np.divide(self.sum_rewards, np.maximum(1, self.counts))


##########################################################################
## SelfishRobustMMABPlayer
##########################################################################
class SelfishRobustMMABPlayer:
    """
    A collision-averse baseline player. This class uses a 'penalty' approach
    to discourage arms with high collision rates, as players receive a fraction
    of the arm's reward. Key idea is to detect collisions if the observed reward
    is significantly below the player's empirical mean.

    This is more robust than a naive approach but can still be suboptimal
    compared to the SMAA method, since it doesn't explicitly compute the
    equilibrium or share arms optimally.
    """

    def __init__(self, player_id, K, T, seed=None, N=None, tau=0.01):
        """
        Args:
            player_id (int): Unique ID of the player.
            K (int): Number of arms.
            T (int): Total rounds.
            seed (int): Random seed for reproducibility.
            N (int): Optional number of players.
            tau (float): Softmax temperature; lower tau => less random exploration.
        """
        self.player_id = player_id
        self.K = K
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.tau = tau

        self.counts = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K)
        self.history = []
        self.collision_counts = np.zeros(K, dtype=int)

    def initialize(self, t):
        """
        Early-phase choice of arm, can be random or round-robin.

        Args:
            t (int): Current round index.

        Returns:
            int: Chosen arm index.
        """
        arm = self.rng.integers(0, self.K)
        self.history.append(arm)
        return arm

    def choose_arm(self, t):
        """
        Chooses an arm based on a weighted softmax of (empirical_mean - penalty),
        where the penalty is proportional to collision_counts.

        Args:
            t (int): Current round index.

        Returns:
            int: The chosen arm index.
        """
        adjusted_values = []
        for k in range(self.K):
            mean_k = self._empirical_mean(k)
            # Penalty fraction ~ fraction of collisions
            penalty_fraction = self.collision_counts[k] / max(1, self.counts[k])
            adjusted_val = max(0.0, mean_k - 0.5 * penalty_fraction)
            adjusted_values.append(adjusted_val)

        exps = np.exp(np.array(adjusted_values) / self.tau)
        probs = exps / np.sum(exps)

        arm = self.rng.choice(self.K, p=probs)
        self.history.append(arm)
        return arm

    def update(self, chosen_arm, total_reward):
        """
        Updates internal stats. If total_reward is much smaller than the current mean,
        we treat it as a sign of collision.

        Args:
            chosen_arm (int): Index of the pulled arm.
            total_reward (float): The entire reward from that arm.
        """
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += total_reward

        cmean = self._empirical_mean(chosen_arm)
        if total_reward < 0.7 * cmean:
            self.collision_counts[chosen_arm] += 1

    def _empirical_mean(self, arm):
        """
        Returns the empirical mean reward for a given arm.

        Args:
            arm (int): Arm index.

        Returns:
            float: Average reward for arm 'arm'.
        """
        return self.sum_rewards[arm] / max(1, self.counts[arm])

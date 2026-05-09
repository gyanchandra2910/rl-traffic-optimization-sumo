"""
Tabular Q-Learning Agent for 6D Macroscopic Traffic State.

Context in Project: "Traffic Signal Optimization using RL"
Implements the foundational model-free TD-learning algorithm using a highly 
efficient discrete state discretization matrix.
"""

import numpy as np
import random
import pickle
import os


class QLearningAgent:
    """
    Improved Tabular Q-Learning Agent.

    Mathematical Intuition (Why Tabular?):
    In massive continuous state spaces, Q-learning fails because tables grow
    infinitely large. However, we cleverly engineered our state space to be a
    6D macroscopic abstraction (Queue Bins, Wait Bins, etc). Because this
    discretized state space is so small, Tabular Q-Learning actually learns
    much faster than Deep Q-Networks due to having ZERO computational overhead 
    from backpropagation.

    The primary downside of Standard Q-Learning is "Overestimation Bias",
    which we address later via Double Q-Learning.
    """

    def __init__(self, action_dim: int = 5, lr: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the Tabular Q-Learning parameters.

        Args:
            action_dim (int): Number of actions (5).
            lr (float): Learning rate (alpha). High = fast updates.
            gamma (float): Discount factor. Weight placed on future vs immediate rewards.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Multiplier per episode to shrink epsilon.
        """
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # The Q-Table dictionary mapping tuple(discrete_state) -> array[actions]
        self.q_table = {}

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Convert the continuous 6-dim macro state to a discrete tuple representation.

        Why Discretization?:
        A queue of 3 and a queue of 4 represent the exact same semantic reality 
        ("Small Traffic"). By binning these floats into discrete integer groups,
        the agent encounters the "same" state more often, massively improving
        sample efficiency and convergence speed.

        Args:
            state (np.ndarray): 6-dim continuous state.

        Returns:
            tuple: Discrete state for Q-table key.
        """
        s = np.array(state, dtype=np.float32).flatten()
        if len(s) < 6:
            padded = np.zeros(6, dtype=np.float32)
            padded[:len(s)] = s
            s = padded

        ns_queue, ew_queue, ns_wait, ew_wait, phase_raw, elapsed = s[:6]

        def queue_bin(value: float) -> int:
            if value <= 0: return 0
            if value <= 5: return 1     # Light traffic
            if value <= 15: return 2    # Moderate congestion
            return 3                    # Heavy congestion

        def wait_bin(value: float) -> int:
            if value <= 20: return 0    # Acceptable wait
            if value <= 80: return 1    # Irritating wait
            return 2                    # Unacceptable wait (force switch)

        phase = int(np.clip(int(phase_raw), 0, 3))

        if elapsed <= 15: ebin = 0
        elif elapsed <= 40: ebin = 1
        else: ebin = 2

        return (
            queue_bin(ns_queue), queue_bin(ew_queue),
            wait_bin(ns_wait), wait_bin(ew_wait),
            phase, ebin,
        )

    def get_q_values(self, discrete_state: tuple) -> np.ndarray:
        """Fetch Q-values dynamically. Zero-initialize unseen states."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_dim)
        return self.q_table[discrete_state]

    def choose_action(self, state: np.ndarray) -> int:
        """
        Standard epsilon-greedy policy execution.
        With probability epsilon, explore. Otherwise, Exploit (argmax).
        """
        discrete_state = self._discretize_state(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            q_values = self.get_q_values(discrete_state)
            return int(np.argmax(q_values))

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray):
        """
        Execute the core Bellman Temporal Difference update.

        Formula:
        Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        ds = self._discretize_state(state)
        dns = self._discretize_state(next_state)

        current_q = self.get_q_values(ds)[action]
        max_next_q = np.max(self.get_q_values(dns))

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[ds][action] = current_q + self.lr * td_error

    def decay_epsilon(self):
        """Exponential decay applied at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Persist table dictionary via Pickle."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath: str):
        """Load table dictionary via Pickle."""
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)

    def get_stats(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'lr': self.lr,
            'gamma': self.gamma
        }

"""
Fixed-Time Traffic Light Controller (Baseline)

Context in Project: "Traffic Signal Optimization using RL"
Implements a non-learning, purely rule-based traffic light controller.
It establishes the fundamental baseline against which all Deep RL and Tabular
agents are compared in the thesis.
"""

import numpy as np


class FixedTimeAgent:
    """
    Fixed-Time (Periodic) Traffic Light Controller.

    Mathematical Intuition (Why it fails):
    A fixed-time controller operates blindly. It uses a static phase order
    (NS-straight → NS-left → EW-straight → EW-left) and gives exactly 30s
    of green light to each phase. 
    
    If the North-South road is completely empty but the East-West road is 
    jammed, the fixed-time controller will still force the East-West traffic 
    to wait for 60+ seconds. This inability to perceive state dynamically
    leads to severe congestion and massive delays.

    This agent conforms to the same `choose_action` interface as the neural 
    networks to allow a perfectly fair comparison loop in `run_final_300.py`.
    """

    def __init__(self, action_dim: int = 5, green_duration: int = 30,
                 delta_time: int = 10):
        """
        Initialize the Fixed-Time controller.

        Args:
            action_dim (int): Number of actions (for interface compatibility).
            green_duration (int): Green phase duration in seconds. Default 30s.
            delta_time (int): Seconds per RL step. Default 10.
        """
        self.action_dim = action_dim
        self.green_duration = green_duration
        self.delta_time = delta_time

        # Calculate how many environment steps equal one full green phase
        self.steps_per_phase = max(1, green_duration // delta_time)

        # Static Phase order: 0→1→2→3→0→...
        self.phase_order = [0, 1, 2, 3]
        self.current_phase_idx = 0
        self.steps_in_phase = 0

        # Dummy attributes to prevent pipeline errors when comparing against RL agents
        self.epsilon = 0.0
        self.lr = 0.0
        self.gamma = 0.0

    def choose_action(self, state: np.ndarray) -> int:
        """
        Select action strictly based on the internal clock, ignoring state.

        Args:
            state (np.ndarray): Current state (completely ignored).

        Returns:
            int: Action (0 = keep, 1-4 = switch to specific phase).
        """
        self.steps_in_phase += 1

        if self.steps_in_phase >= self.steps_per_phase:
            # The static timer expired. Switch to the next logical phase.
            self.current_phase_idx = (self.current_phase_idx + 1) % 4
            self.steps_in_phase = 0
            # Return action to switch (action 0 is "keep", 1-4 are switches)
            return self.phase_order[self.current_phase_idx] + 1
        else:
            # Not enough time has passed. Force the green light to hold.
            return 0

    def learn(self, state, action, reward, next_state):
        """No learning occurs. This is a non-adaptive baseline."""
        pass

    def decay_epsilon(self):
        """No exploration occurs. This is a deterministic baseline."""
        pass

    def save(self, filepath: str):
        """No model weights to serialize."""
        pass

    def load(self, filepath: str):
        """No model weights to deserialize."""
        pass

    def get_stats(self) -> dict:
        """Fetch internal timer statistics for debugging."""
        return {
            'type': 'fixed_time',
            'green_duration': self.green_duration,
            'current_phase_idx': self.current_phase_idx,
            'steps_in_phase': self.steps_in_phase,
        }

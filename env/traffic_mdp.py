"""
Traffic Light MDP Definition (Macroscopic 6D State)

Context in Project: "Traffic Signal Optimization using RL"
This module mathematically defines the Markov Decision Process (MDP) for the
four-arm traffic intersection. It encapsulates the State Space, Action Space,
and Reward Function logic independently from the SUMO simulator to maintain
a clean RL architecture.
"""

import numpy as np


class TrafficMDP:
    """
    Markov Decision Process (MDP) for Traffic Light Control.

    This class provides the core mathematical formulation for the agent's interaction
    with the traffic intersection. By abstracting the state into a 6D macroscopic 
    representation, we avoid the exponential state-space explosion typically associated
    with grid-based vehicle tracking, allowing tabular and small neural networks
    to converge rapidly on CPU architectures.

    State Space (6-dimensional continuous vector):
        - NS_queue: Total halted vehicles on North+South incoming arms.
        - EW_queue: Total halted vehicles on East+West incoming arms.
        - NS_wait: Aggregate waiting time on North+South incoming arms.
        - EW_wait: Aggregate waiting time on East+West incoming arms.
        - current_phase: Active green phase index (0-3).
        - elapsed_phase_time: Seconds the current phase has been active.

    Action Space (5 discrete actions):
        - 0: Keep the current phase (no switch).
        - 1: Switch to NS-straight green.
        - 2: Switch to NS-left-turn green.
        - 3: Switch to EW-straight green.
        - 4: Switch to EW-left-turn green.

    Reward Function:
        R = -α·ΔW - β·Q - γ·C + δ·T
        We penalize waiting time increases and queues, penalize switches, and
        reward vehicle throughput.
    """

    # Reward weights derived from our extensive "Reward Function Ablation Study"
    # Scenario 4 ("Wait Focused") dominated the results with Throughput=217 and Penalty=-1144.80.
    # Intuition:
    # - Waiting time (0.8) is heavily penalized to force the agent to prioritize quality of flow.
    # - Queue length (0.2) is less penalized because large queues are acceptable if they are moving quickly.
    # - Phase changes (2.0) are heavily penalized to prevent flickering lights and maintain realistic traffic stability.
    # - Throughput (1.0) provides a positive signal for clearing the intersection.
    REWARD_WEIGHT_DELTA_WAIT = 0.8    # α: Heavily penalize waiting time
    REWARD_WEIGHT_QUEUE = 0.2         # β: Lightly penalize queue accumulation
    REWARD_WEIGHT_PHASE_CHANGE = 2.0  # γ: Strongly penalize unnecessary phase switching (stability)
    REWARD_WEIGHT_THROUGHPUT = 1.0    # δ: Reward vehicles completing their journey

    # Phase definitions for a 4-way intersection
    PHASE_NS_STRAIGHT = 0   # North-South straight green
    PHASE_NS_LEFT = 1       # North-South left turn green
    PHASE_EW_STRAIGHT = 2   # East-West straight green
    PHASE_EW_LEFT = 3       # East-West left turn green

    NUM_GREEN_PHASES = 4

    # Yellow phase transition duration (seconds) required for realism
    YELLOW_DURATION = 4

    # Minimum green phase duration before allowing a switch (seconds)
    MIN_GREEN_DURATION = 10

    # Maximum green phase duration before forcing a switch (seconds)
    MAX_GREEN_DURATION = 60

    def __init__(self):
        """
        Initialize the Traffic Light MDP boundaries.
        """
        # 4-way intersection with 2 incoming lanes each
        self.num_lanes = 8

        # Macroscopic state: NS/EW queue + NS/EW wait + phase + elapsed
        self.state_dim = 6
        self.state_shape = (self.state_dim,)

        # Action space: keep + 4 green phases
        self.action_dim = 5
        self.action_space = list(range(self.action_dim))

    def calculate_reward(self, total_queue: float, delta_waiting: float,
                         throughput: float, phase_changed: bool) -> float:
        """
        Calculate the multi-objective reward for the current timestep.

        The mathematical intuition here is to provide a dense, immediate reward signal.
        Instead of rewarding the agent only when a vehicle leaves, penalizing the
        derivative of waiting time (delta_waiting) immediately punishes the agent
        for letting vehicles sit idle.

        Args:
            total_queue (float): Sum of all halting vehicles.
            delta_waiting (float): Increase in total waiting time from the previous step.
            throughput (float): Vehicles that reached their destination in this step.
            phase_changed (bool): True if the agent triggered a phase transition.

        Returns:
            float: The calculated reward scalar.
        """
        reward = (
            -self.REWARD_WEIGHT_DELTA_WAIT * delta_waiting
            - self.REWARD_WEIGHT_QUEUE * total_queue
            - self.REWARD_WEIGHT_PHASE_CHANGE * float(phase_changed)
            + self.REWARD_WEIGHT_THROUGHPUT * throughput
        )
        return float(reward)

    def is_action_valid(self, action: int, current_phase: int,
                        elapsed_time: float) -> bool:
        """
        Check if an action respects traffic light safety constraints.

        Why this matters:
        We cannot allow the agent to switch phases every second (flickering).
        It must respect the MIN_GREEN_DURATION constraint.

        Args:
            action (int): Proposed action (0-4).
            current_phase (int): Currently active phase (0-3).
            elapsed_time (float): Seconds the current phase has been active.

        Returns:
            bool: True if the action is legal.
        """
        if action == 0:
            return True  # 'Keep' is always legal

        target_phase = action - 1

        if target_phase == current_phase:
            return True  # Switching to the same phase is just a 'keep'

        # Block switch if minimum green time hasn't passed
        if elapsed_time < self.MIN_GREEN_DURATION:
            return False

        return True

    def get_target_phase(self, action: int, current_phase: int) -> int:
        """
        Convert the raw discrete action space (0-4) into an actual phase index (0-3).

        Args:
            action (int): The selected action (0-4).
            current_phase (int): The current active phase (0-3).

        Returns:
            int: The resulting target phase index.
        """
        if action == 0:
            return current_phase
        return action - 1

    def describe(self) -> str:
        """Return a human-readable description of the configured MDP parameters."""
        return (
            f"TrafficMDP:\n"
            f"  State dim: {self.state_dim} (NS_queue, EW_queue, NS_wait, EW_wait, phase, elapsed)\n"
            f"  Action dim: {self.action_dim} (keep + 4 green phases)\n"
            f"  Reward Weights:\n"
            f"    Wait = {self.REWARD_WEIGHT_DELTA_WAIT}\n"
            f"    Queue = {self.REWARD_WEIGHT_QUEUE}\n"
            f"    Phase_Chg = {self.REWARD_WEIGHT_PHASE_CHANGE}\n"
            f"    Throughput = {self.REWARD_WEIGHT_THROUGHPUT}\n"
        )

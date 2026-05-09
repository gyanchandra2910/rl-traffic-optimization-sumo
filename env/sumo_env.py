"""
SUMO Environment Wrapper for Reinforcement Learning.

Context in Project: "Traffic Signal Optimization using RL"
This class constructs a bridge between the RL algorithms (acting as the brain)
and Eclipse SUMO (acting as the physical simulation). It abstracts away
complex TraCI commands and provides a clean, OpenAI Gym-like interface.
"""

import os
import sys
import numpy as np

# Check for SUMO_HOME environment variable to ensure TraCI imports correctly
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

from .traffic_mdp import TrafficMDP


class SumoEnv:
    """
    SUMO Environment Wrapper for Reinforcement Learning.

    Why this is critical:
    Instead of dealing with raw SUMO junctions, this wrapper distills the
    traffic flow into a 6D mathematical macroscopic state, enabling
    algorithms like PPO and TRPO to learn efficiently.

    Key features:
        - Realistic yellow-phase transitions (safety delays).
        - Multi-objective reward integration.
        - Granular per-step metric extraction for thesis reporting.
    """

    # Logical phase groups for 2-lane left-hand traffic (Indian style)
    # lane 0: straight+right, lane 1: protected left
    PHASE_DEFS = {
        0: {"name": "NS_STRAIGHT_RIGHT", "arm_group": "NS", "movement": "SR"},
        1: {"name": "NS_LEFT", "arm_group": "NS", "movement": "LEFT"},
        2: {"name": "EW_STRAIGHT_RIGHT", "arm_group": "EW", "movement": "SR"},
        3: {"name": "EW_LEFT", "arm_group": "EW", "movement": "LEFT"},
    }

    def __init__(self, net_file: str, route_file: str, use_gui: bool = False,
                 sim_max_steps: int = 3600, delta_time: int = 10):
        """
        Initialize the SUMO environment parameters.

        Args:
            net_file (str): Path to the SUMO network file.
            route_file (str): Path to the SUMO route file.
            use_gui (bool): If True, use sumo-gui. Otherwise uses sumo (faster).
            sim_max_steps (int): Total seconds per training episode.
            delta_time (int): Seconds elapsed per individual RL step decision.
        """
        self.mdp = TrafficMDP()
        self.state_size = self.mdp.state_dim
        self.action_size = self.mdp.action_dim  # 5 actions

        self.net_file = net_file
        self.route_file = route_file

        data_dir = os.path.dirname(net_file)
        self.vtypes_file = os.path.join(data_dir, "vtypes.add.xml")

        self.use_gui = use_gui
        self.sim_max_steps = sim_max_steps
        self.delta_time = delta_time
        self.current_step = 0

        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        # Command arguments optimized for speed and headless running
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.vtypes_file,
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]

        if self.use_gui:
            self.sumo_cmd.extend(["--start", "--quit-on-end"])

        # 8 incoming lanes for our 4-arm left-hand intersection
        self.lanes = [
            "B1A1_0", "B1A1_1",
            "B2A1_0", "B2A1_1",
            "B3A1_0", "B3A1_1",
            "B4A1_0", "B4A1_1",
        ]

        self.ns_arms = {"B1A1", "B3A1"}
        self.ew_arms = {"B2A1", "B4A1"}
        self.signal_groups = []

        self.tl_id = "A1"

        self.current_phase = 0
        self.elapsed_phase_time = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.next_phase = 0

        self.prev_total_waiting_time = 0.0
        self.total_vehicles_arrived = 0

    def reset(self) -> np.ndarray:
        """
        Reset the simulation to absolute zero.

        Closes any old TraCI socket connections gracefully before spinning
        up a fresh simulation instance.

        Returns:
            np.ndarray: The initial (usually zeroed) 6D state array.
        """
        try:
            traci.close()
        except Exception:
            pass

        traci.start(self.sumo_cmd)

        self.current_step = 0
        self.current_phase = 0
        self.elapsed_phase_time = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.next_phase = 0
        self.prev_total_waiting_time = 0.0
        self.total_vehicles_arrived = 0

        self._initialize_signal_groups()
        self._set_green_phase(0)

        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Execute an RL action and step the SUMO physics engine forward.

        Why Yellow Transitions Matter:
        If an agent switches the light from NS to EW, we cannot instantly turn
        the lights green, or cars will crash in the simulation. We inject a
        mandatory 4-second Yellow phase during which the RL agent must simply
        wait. This penalizes the agent for switching phases too often.

        Args:
            action (int): The chosen decision (0-4).

        Returns:
            tuple: (next_state, reward, done, info_dict)
        """
        phase_changed = False

        if not self.is_yellow:
            target_phase = self.mdp.get_target_phase(action, self.current_phase)

            if target_phase != self.current_phase:
                if self.elapsed_phase_time >= self.mdp.MIN_GREEN_DURATION:
                    # Trigger the yellow safety mechanism
                    self.is_yellow = True
                    self.yellow_timer = 0
                    self.next_phase = target_phase
                    self._set_yellow_phase()
                    phase_changed = True

        # Advance the underlying SUMO engine by `delta_time`
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.current_step += 1
            self.elapsed_phase_time += 1

            if self.is_yellow:
                self.yellow_timer += 1
                if self.yellow_timer >= self.mdp.YELLOW_DURATION:
                    self.current_phase = self.next_phase
                    self.elapsed_phase_time = 0
                    self.is_yellow = False
                    self.yellow_timer = 0
                    self._set_green_phase(self.current_phase)

        metrics = self._compute_metrics()

        # Reward is dense: It immediately penalizes the derivative of waiting time
        delta_waiting = metrics['total_waiting_time'] - self.prev_total_waiting_time
        reward = self.mdp.calculate_reward(
            total_queue=metrics['total_queue_length'],
            delta_waiting=delta_waiting,
            throughput=metrics['step_throughput'],
            phase_changed=phase_changed
        )

        self.prev_total_waiting_time = metrics['total_waiting_time']
        self.total_vehicles_arrived += metrics['step_throughput']

        next_state = self._get_state()
        done = self.current_step >= self.sim_max_steps

        # Info dict used heavily for thesis metric plotting
        info = {
            'total_queue_length': metrics['total_queue_length'],
            'total_waiting_time': metrics['total_waiting_time'],
            'avg_waiting_time': metrics['avg_waiting_time'],
            'step_throughput': metrics['step_throughput'],
            'total_throughput': self.total_vehicles_arrived,
            'avg_delay': metrics['avg_delay'],
            'phase_changed': phase_changed,
            'current_phase': self.current_phase,
            'queue_per_lane': metrics['queue_per_lane'],
            'wait_per_lane': metrics['wait_per_lane'],
            'num_vehicles': metrics['num_vehicles'],
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Construct the 6D Macroscopic State vector via TraCI sensors."""
        ns_queue = 0.0
        ew_queue = 0.0
        ns_wait = 0.0
        ew_wait = 0.0

        for lane in self.lanes:
            q = traci.lane.getLastStepHaltingNumber(lane)
            w = traci.lane.getWaitingTime(lane)
            arm = lane.split('_')[0]
            if arm in self.ns_arms:
                ns_queue += q
                ns_wait += w
            else:
                ew_queue += q
                ew_wait += w

        state = np.array(
            [
                float(ns_queue),
                float(ew_queue),
                float(ns_wait),
                float(ew_wait),
                float(self.current_phase),
                float(self.elapsed_phase_time),
            ],
            dtype=np.float32,
        )
        return state

    def _compute_metrics(self) -> dict:
        """Internal routine to calculate metrics for reporting."""
        ns_queue = 0
        ew_queue = 0
        ns_wait = 0.0
        ew_wait = 0.0
        queue_per_lane = []
        wait_per_lane = []

        for lane in self.lanes:
            q = traci.lane.getLastStepHaltingNumber(lane)
            w = traci.lane.getWaitingTime(lane)
            arm = lane.split('_')[0]
            if arm in self.ns_arms:
                ns_queue += q
                ns_wait += w
            else:
                ew_queue += q
                ew_wait += w
            queue_per_lane.append(q)
            wait_per_lane.append(w)

        total_queue = ns_queue + ew_queue
        total_wait = ns_wait + ew_wait
        step_throughput = traci.simulation.getArrivedNumber()
        num_vehicles = traci.vehicle.getIDCount()
        avg_waiting = total_wait / max(num_vehicles, 1)

        avg_delay = 0.0
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) > 0:
            total_delay = 0.0
            for vid in vehicle_ids:
                total_delay += traci.vehicle.getWaitingTime(vid)
            avg_delay = total_delay / len(vehicle_ids)

        return {
            'total_queue_length': total_queue,
            'total_waiting_time': total_wait,
            'avg_waiting_time': avg_waiting,
            'step_throughput': step_throughput,
            'avg_delay': avg_delay,
            'ns_queue': ns_queue,
            'ew_queue': ew_queue,
            'ns_wait': ns_wait,
            'ew_wait': ew_wait,
            'queue_per_lane': queue_per_lane,
            'wait_per_lane': wait_per_lane,
            'num_vehicles': num_vehicles,
        }

    def _initialize_signal_groups(self):
        """Map SUMO junctions logically."""
        try:
            controlled_links = traci.trafficlight.getControlledLinks(self.tl_id)
            self.signal_groups = []
            for link_set in controlled_links:
                if not link_set:
                    self.signal_groups.append(("UNK", "UNK"))
                    continue

                in_lane = link_set[0][0] if len(link_set[0]) > 0 else ""
                lane_prefix, lane_idx = self._parse_lane_info(in_lane)
                arm_group = "NS" if lane_prefix in self.ns_arms else "EW"
                movement = "LEFT" if lane_idx == 1 else "SR"
                self.signal_groups.append((arm_group, movement))
        except Exception:
            self.signal_groups = self._build_fallback_signal_groups(8)

    def _parse_lane_info(self, lane_id: str):
        try:
            prefix, idx = lane_id.rsplit('_', 1)
            return prefix, int(idx)
        except Exception:
            return lane_id, 0

    def _build_fallback_signal_groups(self, num_signals: int):
        fallback = []
        lanes = self.lanes[:num_signals]
        for lane in lanes:
            lane_prefix, lane_idx = self._parse_lane_info(lane)
            arm_group = "NS" if lane_prefix in self.ns_arms else "EW"
            movement = "LEFT" if lane_idx == 1 else "SR"
            fallback.append((arm_group, movement))
        while len(fallback) < num_signals:
            fallback.append(("UNK", "UNK"))
        return fallback

    def _set_green_phase(self, phase: int):
        """Force TraCI to execute the green phase string."""
        try:
            controlled_links = traci.trafficlight.getControlledLinks(self.tl_id)
            num_signals = len(controlled_links)
        except Exception:
            num_signals = max(len(self.signal_groups), 8)

        state_str = self._build_phase_string(phase, is_yellow=False, num_signals=num_signals)
        traci.trafficlight.setRedYellowGreenState(self.tl_id, state_str)

    def _set_yellow_phase(self):
        """Force TraCI to execute the yellow transition string."""
        try:
            controlled_links = traci.trafficlight.getControlledLinks(self.tl_id)
            num_signals = len(controlled_links)
        except Exception:
            num_signals = max(len(self.signal_groups), 8)

        state_str = self._build_phase_string(self.current_phase, is_yellow=True, num_signals=num_signals)
        traci.trafficlight.setRedYellowGreenState(self.tl_id, state_str)

    def _build_phase_string(self, phase: int, is_yellow: bool, num_signals: int) -> str:
        """Construct the traffic light ASCII strings (e.g. 'GGGrrrGGGrrr')."""
        signals = ['r'] * num_signals

        if not self.signal_groups:
            self.signal_groups = self._build_fallback_signal_groups(num_signals)
        if len(self.signal_groups) < num_signals:
            self.signal_groups.extend(self._build_fallback_signal_groups(num_signals - len(self.signal_groups)))

        phase_def = self.PHASE_DEFS.get(phase, self.PHASE_DEFS[0])
        arm_group = phase_def["arm_group"]
        movement = phase_def["movement"]

        char = 'y' if is_yellow else 'G'
        for idx in range(num_signals):
            group = self.signal_groups[idx] if idx < len(self.signal_groups) else ("UNK", "UNK")
            if group[0] == arm_group and group[1] == movement:
                signals[idx] = char

        return ''.join(signals)

    def close(self):
        """Clean shutdown of TraCI socket."""
        try:
            traci.close()
        except Exception:
            pass

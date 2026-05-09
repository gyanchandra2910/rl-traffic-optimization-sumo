"""
Deep Q-Network (DQN) Agent for Traffic Light Control.

Context in Project: "Traffic Signal Optimization using RL"
Implements standard Deep Q-Networks (Mnih et al., 2015). This agent replaces 
tabular arrays with neural function approximators, allowing it to directly 
ingest continuous, un-discretized inputs.
"""

import numpy as np
import random
import os
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except (ImportError, OSError) as exc:
    TORCH_AVAILABLE = False
    print(f"[WARNING] PyTorch unavailable ({exc}). DQN agent will not be available.")


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Mathematical Intuition (Why it's needed):
    Sequential data in RL (State 1, State 2, State 3) is highly correlated.
    Feeding correlated data directly into a neural network breaks the I.I.D. 
    (Independent and Identically Distributed) assumption of stochastic gradient 
    descent, causing catastrophic forgetting.
    
    The Replay Buffer stores past transitions and randomly samples them during
    training to completely break this temporal correlation.
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """
        Neural Network for Action-Value approximation.
        Fast, lightweight 64x64 CPU-friendly topology.
        """
        def __init__(self, state_dim: int = 6, action_dim: int = 5):
            super(QNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.network(x)


class DQNAgent:
    """
    Deep Q-Network Agent.

    Mathematical Intuition:
    Unlike tabular Q-Learning which discretizes the state space into bins,
    DQN maps the continuous 6D state directly to Q-values using backpropagation.
    
    Target Network Strategy:
    If the agent uses the same network to compute the target value and the
    current prediction, it creates a "chasing a moving target" instability.
    DQN freezes a separate Target Network to compute TD-targets stably, only
    updating its weights every N steps.
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 5,
                 lr: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 50000, batch_size: int = 64,
                 target_update_freq: int = 500):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQN agent.")

        self.state_dim = 6
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core Networks
        self.q_network = QNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for robust gradients against outliers

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0

        # Welford's Running Stats for robust State Normalization
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_std = np.ones(self.state_dim, dtype=np.float32)
        self.state_count = 0

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.size >= self.state_dim:
            return arr[:self.state_dim]
        out = np.zeros(self.state_dim, dtype=np.float32)
        out[:arr.size] = arr
        return out

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Online state normalization using Welford's algorithm.
        CRITICAL: Prevents gradient explosion from massive waiting times (>1000).
        """
        state = self._prepare_state(state)
        self.state_count += 1
        alpha = 1.0 / self.state_count
        
        self.state_mean = (1 - alpha) * self.state_mean + alpha * state
        diff = state - self.state_mean
        self.state_std = np.sqrt(
            (1 - alpha) * (self.state_std ** 2) + alpha * (diff ** 2)
        )

        std = np.maximum(self.state_std, 1e-6)
        return (state - self.state_mean) / std

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy exploration wrapped around neural inference."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        norm_state = self._normalize_state(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store perfectly normalized state experiences."""
        norm_state = self._normalize_state(state)
        norm_next = self._normalize_state(next_state)
        self.replay_buffer.push(norm_state, action, reward, norm_next, done)

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Execute Bellman backpropagation utilizing the Target Network.
        """
        self.store_transition(state, action, reward, next_state, done=False)

        # Guard: Wait until sufficient entropy is built up in buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q: gather the value of the action taken
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Stable Target Q: evaluated purely by the frozen Target Network
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Hard-copy update to the Target Network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dim': self.state_dim,
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count,
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        loaded_mean = checkpoint.get('state_mean', self.state_mean)
        loaded_std = checkpoint.get('state_std', self.state_std)
        if np.asarray(loaded_mean).shape == self.state_mean.shape:
            self.state_mean = loaded_mean
        if np.asarray(loaded_std).shape == self.state_std.shape:
            self.state_std = loaded_std
        self.state_count = checkpoint.get('state_count', 0)
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)

    def get_stats(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'learn_steps': self.learn_step_counter,
        }

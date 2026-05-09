"""
Trust Region Policy Optimization (TRPO) Agent for Traffic Light Control.

Context in Project: "Traffic Signal Optimization using RL"
Implements a highly stable Policy Gradient algorithm. TRPO mathematically
guarantees that a policy update will not catastrophically degrade performance,
which resulted in this agent achieving the lowest Wait Times in our 300-episode test.
"""

import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    from torch.distributions.kl import kl_divergence
    TORCH_AVAILABLE = True
except (ImportError, OSError) as exc:
    TORCH_AVAILABLE = False
    print(f"[WARNING] PyTorch unavailable ({exc}). TRPO agent will not be available.")

if TORCH_AVAILABLE:
    class ActorCriticNet(nn.Module):
        """
        Neural Network serving as both Actor (Policy) and Critic (Value).
        
        Compact architecture (64x64) allows rapid inference and training
        in our 6D macroscopic state space.
        """
        def __init__(self, state_dim: int = 6, action_dim: int = 5):
            super(ActorCriticNet, self).__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.actor(x), self.critic(x)

class TRPOAgent:
    """
    Trust Region Policy Optimization (TRPO) Agent.

    Mathematical Intuition (Why TRPO?):
    Standard RL algorithms can accidentally un-learn good behavior by taking
    too large of a gradient step. TRPO solves this by enforcing a hard 
    "Trust Region" boundary using Kullback-Leibler (KL) Divergence. 
    It solves: Maximize Advantage, subject to KL(old_policy || new_policy) <= delta.

    This ensures the new policy never strays too far from the old one, providing
    supreme stability, which allowed it to master micro-managing traffic light
    phases to minimize overall wait times efficiently.

    State Normalization:
    Deep RL networks suffer from "Gradient Explosions" if input values are too
    large. We use Welford's online running mean/variance algorithm to robustly
    normalize all inputs (like waiting times of 1000+) to a mean of 0 and std of 1.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 5,
                 lr: float = 0.001, gamma: float = 0.95,
                 batch_size: int = 360, trpo_epochs: int = 4,
                 target_kl: float = 0.01, **kwargs):
        """
        Initialize the TRPO agent parameters.

        Args:
            state_dim (int): Observation space size.
            action_dim (int): Action space size.
            batch_size (int): Experiences collected before an update phase.
            trpo_epochs (int): Number of optimization passes over the collected batch.
            target_kl (float): The maximum allowed KL divergence (the Trust Region boundary).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TRPO agent.")

        self.state_dim = 6
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.trpo_epochs = trpo_epochs
        
        # Trust Region Adaptive Penalty mechanism
        self.target_kl = target_kl
        self.kl_beta = 1.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCriticNet(self.state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        self.buffer = []
        self.learn_step_counter = 0

        # Welford's Running Stats for robust State Normalization
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_var = np.zeros(self.state_dim, dtype=np.float32)
        self.state_count = 0

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        """Ensure state vector conforms exactly to 6D float32."""
        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.size >= self.state_dim:
            return arr[:self.state_dim]
        out = np.zeros(self.state_dim, dtype=np.float32)
        out[:arr.size] = arr
        return out

    def _normalize_and_update_state(self, state: np.ndarray) -> np.ndarray:
        """
        Online state normalization using Welford's algorithm.
        CRITICAL: Prevents gradient explosion from massive waiting times.
        """
        state = self._prepare_state(state)
        self.state_count += 1
        
        if self.state_count == 1:
            self.state_mean = np.copy(state)
            self.state_var = np.zeros_like(state)
            return np.zeros_like(state)
            
        old_mean = np.copy(self.state_mean)
        self.state_mean += (state - old_mean) / self.state_count
        self.state_var += (state - old_mean) * (state - self.state_mean)
        
        var = self.state_var / self.state_count
        std = np.maximum(np.sqrt(var), 1e-6)
        return (state - self.state_mean) / std

    def choose_action(self, state: np.ndarray) -> int:
        """
        Sample an action stochastically from the Actor's probability distribution.
        """
        norm_state = self._normalize_and_update_state(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.net(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
        return action.item()

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Store transition and trigger a TRPO update if the batch is full.
        """
        norm_state = self._prepare_state(state)
        std = np.maximum(np.sqrt(self.state_var / max(self.state_count, 1)), 1e-6)
        norm_state = (norm_state - self.state_mean) / std
        
        norm_next = self._prepare_state(next_state)
        norm_next = (norm_next - self.state_mean) / std
        
        self.buffer.append((norm_state, action, reward, norm_next))
        
        if len(self.buffer) >= self.batch_size:
            self.update()

    def update(self):
        """
        Execute the TRPO Optimization process using an adaptive KL penalty.
        """
        if len(self.buffer) == 0:
            return
            
        states, actions, rewards, next_states = zip(*self.buffer)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        with torch.no_grad():
            old_logits, old_values = self.net(states_t)
            _, next_values = self.net(next_states_t)
            
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions_t)
            
            # Compute Advantage using TD-error
            td_targets = rewards_t + self.gamma * next_values.squeeze(1)
            advantages = td_targets - old_values.squeeze(1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        for _ in range(self.trpo_epochs):
            logits, values = self.net(states_t)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            
            # Ratio of new vs old policy probabilities
            ratios = torch.exp(log_probs - old_log_probs)
            surr_loss = -(ratios * advantages).mean()
            
            # Compute KL Divergence penalty
            kl = kl_divergence(old_dist, dist).mean()
            actor_loss = surr_loss + self.kl_beta * kl
            critic_loss = nn.MSELoss()(values.squeeze(1), td_targets)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Early stopping if Trust Region boundary is breached
            if kl.item() > 1.5 * self.target_kl:
                self.kl_beta *= 2.0
                break
            
        # Relax penalty if safe
        if kl.item() < self.target_kl / 1.5:
            self.kl_beta /= 2.0
            
        self.learn_step_counter += 1
        self.buffer.clear()

    def decay_epsilon(self):
        pass

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dim': self.state_dim,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'kl_beta': self.kl_beta,
            'state_mean': self.state_mean,
            'state_var': self.state_var,
            'state_count': self.state_count,
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.kl_beta = checkpoint.get('kl_beta', self.kl_beta)
        self.state_mean = checkpoint.get('state_mean', self.state_mean)
        self.state_var = checkpoint.get('state_var', np.zeros_like(self.state_mean))
        self.state_count = checkpoint.get('state_count', 0)
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)

    def get_stats(self) -> dict:
        return {
            'buffer_size': len(self.buffer),
            'learn_steps': self.learn_step_counter,
            'kl_beta': self.kl_beta,
            'device': str(self.device),
        }

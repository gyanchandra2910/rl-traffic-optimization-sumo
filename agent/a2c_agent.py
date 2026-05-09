"""
Advantage Actor-Critic (A2C) Agent for Traffic Light Control.

Context in Project: "Traffic Signal Optimization using RL"
Implements the A2C algorithm using a hybrid approach where an Actor
network learns the policy, and a Critic network evaluates states to 
reduce gradient variance.
"""

import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except (ImportError, OSError) as exc:
    TORCH_AVAILABLE = False
    print(f"[WARNING] PyTorch unavailable ({exc}). A2C agent will not be available.")

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
            logits = self.actor(x)
            value = self.critic(x)
            return logits, value

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent.

    Mathematical Intuition (Why A2C?):
    Standard Policy Gradients use raw rewards (or returns) to update the policy.
    This creates incredibly high variance during training. A2C introduces the
    Advantage function: A(s,a) = Q(s,a) - V(s). By subtracting a learned baseline
    V(s) (the Critic), the agent learns whether an action was better or worse 
    than the *average* action in that state, significantly stabilizing training.

    State Normalization:
    Deep RL networks suffer from "Gradient Explosions" if input values are too
    large. We use Welford's online running mean/variance algorithm to robustly
    normalize all inputs (like waiting times of 1000+) to a mean of 0 and std of 1.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 5,
                 lr: float = 0.001, gamma: float = 0.95,
                 batch_size: int = 128, entropy_coef: float = 0.01,
                 **kwargs):
        """
        Initialize the A2C agent parameters.

        Args:
            state_dim (int): Observation space size.
            action_dim (int): Action space size.
            batch_size (int): Experiences collected before an update phase.
            entropy_coef (float): Coefficient to encourage exploration.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for A2C agent.")

        self.state_dim = 6
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
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
        Store transition and trigger an A2C update if the batch is full.
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
        Execute the Advantage Actor-Critic Optimization process.
        """
        if len(self.buffer) == 0:
            return
            
        states, actions, rewards, next_states = zip(*self.buffer)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Get Current Values and Logits
        logits, values = self.net(states_t)
        _, next_values = self.net(next_states_t)
        
        # Calculate TD Targets and Advantages
        td_targets = rewards_t + self.gamma * next_values.squeeze(1).detach()
        td_errors = td_targets - values.squeeze(1) # This is the Advantage A(s,a)
        
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        # Actor Loss: Push probabilities towards actions with high Advantage
        actor_loss = -(log_probs * td_errors.detach()).mean()
        # Critic Loss: Minimize prediction error
        critic_loss = nn.SmoothL1Loss()(values.squeeze(1), td_targets)
        
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
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
            'state_mean': self.state_mean,
            'state_var': self.state_var,
            'state_count': self.state_count,
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.state_mean = checkpoint.get('state_mean', self.state_mean)
        self.state_var = checkpoint.get('state_var', np.zeros_like(self.state_mean))
        self.state_count = checkpoint.get('state_count', 0)
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)

    def get_stats(self) -> dict:
        return {
            'buffer_size': len(self.buffer),
            'learn_steps': self.learn_step_counter,
            'device': str(self.device),
        }

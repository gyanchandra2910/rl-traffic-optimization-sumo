"""
Traffic Light Management System - Unified Training Script

Trains any of the four traffic light control agents:
    - Q-Learning (tabular)
    - Double Q-Learning (tabular, reduced overestimation)
    - DQN (neural network, deep RL)
    - Fixed-Time (baseline, no training needed)

Usage:
    python train.py --agent q_learning --episodes 100
    python train.py --agent double_q --episodes 100
    python train.py --agent dqn --episodes 100
    python train.py --agent fixed_time --episodes 10
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from env.sumo_env import SumoEnv
from agent.q_agent import QLearningAgent
from agent.double_q_agent import DoubleQLearningAgent
from agent.dqn_agent import DQNAgent
from agent.fixed_time_agent import FixedTimeAgent


# Agent display names for pretty printing
AGENT_NAMES = {
    'q_learning': 'Q-Learning',
    'double_q': 'Double Q-Learning',
    'dqn': 'Deep Q-Network (DQN)',
    'fixed_time': 'Fixed-Time Controller',
}


def create_agent(agent_type: str):
    """
    Create an agent instance based on agent type string.

    Args:
        agent_type: One of 'q_learning', 'double_q', 'dqn', 'fixed_time'.

    Returns:
        Agent instance.
    """
    if agent_type == 'q_learning':
        return QLearningAgent(
            action_dim=5,
            lr=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )
    elif agent_type == 'double_q':
        return DoubleQLearningAgent(
            action_dim=5,
            lr=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )
    elif agent_type == 'dqn':
        return DQNAgent(
            state_dim=26,
            action_dim=5,
            lr=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_size=50000,
            batch_size=64,
            target_update_freq=500,
        )
    elif agent_type == 'fixed_time':
        return FixedTimeAgent(
            action_dim=5,
            green_duration=30,
            delta_time=10,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def get_model_path(script_dir: str, agent_type: str) -> str:
    """Get the model save path for an agent type."""
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    extensions = {
        'q_learning': 'pkl',
        'double_q': 'pkl',
        'dqn': 'pt',
        'fixed_time': 'none',
    }
    ext = extensions.get(agent_type, 'pkl')
    return os.path.join(models_dir, f"{agent_type}_model.{ext}")


def main():
    """Main training loop supporting all agent types."""

    parser = argparse.ArgumentParser(description='Train Traffic Light RL Agent')
    parser.add_argument('--agent', type=str, default='q_learning',
                        choices=['q_learning', 'double_q', 'dqn', 'fixed_time'],
                        help='Agent type to train (default: q_learning)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--steps', type=int, default=360,
                        help='Max steps per episode (default: 360)')
    parser.add_argument('--gui', action='store_true',
                        help='Use SUMO GUI for visualization')
    args = parser.parse_args()

    # =========================================================================
    # Setup Paths
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(script_dir, "data", "network.net.xml")
    route_file = os.path.join(script_dir, "data", "routes.rou.xml")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Verify files exist
    for f, label in [(net_file, "Network"), (route_file, "Route")]:
        if not os.path.exists(f):
            print(f"[ERROR] {label} file not found: {f}")
            print("Please run 'python utils/generate_sumo_files.py' first.")
            return

    agent_name = AGENT_NAMES.get(args.agent, args.agent)
    print("=" * 60)
    print(f"Training: {agent_name}")
    print("=" * 60)
    print(f"Network file: {net_file}")
    print(f"Route file:   {route_file}")
    print(f"Episodes:     {args.episodes}")
    print(f"Steps/ep:     {args.steps}")

    # =========================================================================
    # Initialize Components
    # =========================================================================
    env = SumoEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
        sim_max_steps=3600,
        delta_time=10,
    )

    agent = create_agent(args.agent)
    model_path = get_model_path(script_dir, args.agent)

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")

    # Metrics tracking
    episode_rewards = []
    episode_avg_queues = []
    episode_avg_waits = []
    episode_throughputs = []
    episode_avg_delays = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        step_queues = []
        step_waits = []
        step_throughputs = []
        step_delays = []

        for step in range(args.steps):
            # Agent selects action
            action = agent.choose_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Agent learns
            agent.learn(state, action, reward, next_state)

            # Track metrics
            state = next_state
            total_reward += reward
            step_queues.append(info.get('total_queue_length', 0))
            step_waits.append(info.get('avg_waiting_time', 0))
            step_throughputs.append(info.get('step_throughput', 0))
            step_delays.append(info.get('avg_delay', 0))

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_avg_queues.append(np.mean(step_queues) if step_queues else 0)
        episode_avg_waits.append(np.mean(step_waits) if step_waits else 0)
        episode_throughputs.append(sum(step_throughputs))
        episode_avg_delays.append(np.mean(step_delays) if step_delays else 0)

        # Print progress
        stats = agent.get_stats()
        eps_str = f"ε={stats.get('epsilon', 0):.4f}" if 'epsilon' in stats else ""
        print(f"Episode {episode+1:3d}/{args.episodes} | "
              f"Reward: {total_reward:10.2f} | "
              f"AvgQ: {episode_avg_queues[-1]:6.1f} | "
              f"AvgWait: {episode_avg_waits[-1]:7.1f} | "
              f"Throughput: {episode_throughputs[-1]:4d} | "
              f"{eps_str}")

    # =========================================================================
    # Cleanup & Save
    # =========================================================================
    env.close()
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    # Save model
    if args.agent != 'fixed_time':
        agent.save(model_path)
        print(f"[INFO] Model saved: {model_path}")

    # Save training metrics to JSON
    metrics = {
        'agent_type': args.agent,
        'episodes': args.episodes,
        'episode_rewards': episode_rewards,
        'episode_avg_queues': episode_avg_queues,
        'episode_avg_waits': episode_avg_waits,
        'episode_throughputs': [int(x) for x in episode_throughputs],
        'episode_avg_delays': episode_avg_delays,
    }
    metrics_path = os.path.join(results_dir, f"{args.agent}_training.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Training metrics saved: {metrics_path}")

    # =========================================================================
    # Plot Training Curves
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{agent_name} — Training Progress', fontsize=16, fontweight='bold')
    episodes_range = range(1, args.episodes + 1)

    # Rewards
    ax = axes[0, 0]
    ax.plot(episodes_range, episode_rewards, color='#2196F3', linewidth=1.5, alpha=0.7)
    if len(episode_rewards) >= 10:
        window = min(10, len(episode_rewards))
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, args.episodes + 1), smoothed, color='#F44336',
                linewidth=2, label=f'{window}-ep moving avg')
        ax.legend()
    ax.set_title('Total Reward per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    # Queue Length
    ax = axes[0, 1]
    ax.plot(episodes_range, episode_avg_queues, color='#FF9800', linewidth=1.5, alpha=0.7)
    ax.set_title('Average Queue Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Queue Length')
    ax.grid(True, alpha=0.3)

    # Waiting Time
    ax = axes[1, 0]
    ax.plot(episodes_range, episode_avg_waits, color='#9C27B0', linewidth=1.5, alpha=0.7)
    ax.set_title('Average Waiting Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Wait (s)')
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1, 1]
    ax.plot(episodes_range, episode_throughputs, color='#4CAF50', linewidth=1.5, alpha=0.7)
    ax.set_title('Total Throughput per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Vehicles')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{args.agent}_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training plot saved: {plot_path}")

    print("\n[SUCCESS] Training completed successfully!")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluate.py --agent {args.agent} --episodes 5")


if __name__ == "__main__":
    main()

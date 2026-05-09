"""
Traffic Light Management System - Evaluation Script

Evaluates any trained agent (or fixed-time baseline) on the SUMO
environment and collects detailed per-step metrics for comparison.

Usage:
    python evaluate.py --agent q_learning --episodes 5
    python evaluate.py --agent double_q --episodes 5
    python evaluate.py --agent dqn --episodes 5
    python evaluate.py --agent fixed_time --episodes 5
    python evaluate.py --agent q_learning --gui  (visualize with SUMO GUI)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.sumo_env import SumoEnv


AGENT_NAMES = {
    'q_learning': 'Q-Learning',
    'double_q': 'Double Q-Learning',
    'dqn': 'Deep Q-Network (DQN)',
    'fixed_time': 'Fixed-Time Controller',
}


def create_agent(agent_type: str):
    """Create an agent for evaluation (low/no exploration)."""
    if agent_type == 'q_learning':
        from agent.q_agent import QLearningAgent
        return QLearningAgent(action_dim=5, epsilon=0.0)
    elif agent_type == 'double_q':
        from agent.double_q_agent import DoubleQLearningAgent
        return DoubleQLearningAgent(action_dim=5, epsilon=0.0)
    elif agent_type == 'dqn':
        from agent.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=26, action_dim=5, epsilon=0.0)
        return agent
    elif agent_type == 'fixed_time':
        from agent.fixed_time_agent import FixedTimeAgent
        return FixedTimeAgent(action_dim=5, green_duration=30, delta_time=10)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def get_model_path(script_dir: str, agent_type: str) -> str:
    """Get the model path for loading."""
    extensions = {
        'q_learning': 'pkl',
        'double_q': 'pkl',
        'dqn': 'pt',
        'fixed_time': 'none',
    }
    ext = extensions.get(agent_type, 'pkl')
    return os.path.join(script_dir, "models", f"{agent_type}_model.{ext}")


def main():
    """Run evaluation for a trained agent."""

    parser = argparse.ArgumentParser(description='Evaluate Traffic Light Agent')
    parser.add_argument('--agent', type=str, default='q_learning',
                        choices=['q_learning', 'double_q', 'dqn', 'fixed_time'],
                        help='Agent type to evaluate')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--steps', type=int, default=360,
                        help='Max steps per episode')
    parser.add_argument('--gui', action='store_true',
                        help='Use SUMO GUI')
    args = parser.parse_args()

    # =========================================================================
    # Setup
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(script_dir, "data", "network.net.xml")
    route_file = os.path.join(script_dir, "data", "routes.rou.xml")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for f, label in [(net_file, "Network"), (route_file, "Route")]:
        if not os.path.exists(f):
            print(f"[ERROR] {label} file not found: {f}")
            return

    agent_name = AGENT_NAMES.get(args.agent, args.agent)
    print("=" * 60)
    print(f"Evaluating: {agent_name}")
    print("=" * 60)

    # =========================================================================
    # Initialize
    # =========================================================================
    env = SumoEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
        sim_max_steps=3600,
        delta_time=10,
    )

    agent = create_agent(args.agent)

    # Load trained model
    model_path = get_model_path(script_dir, args.agent)
    if args.agent != 'fixed_time':
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"[INFO] Loaded model: {model_path}")
        else:
            print(f"[WARNING] No model found at {model_path}. Using untrained agent.")

    # =========================================================================
    # Evaluation Loop
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running Evaluation...")
    print(f"{'='*60}\n")

    all_episode_results = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0

        # Per-step metric lists
        step_queues = []
        step_waits = []
        step_throughputs = []
        step_delays = []
        step_rewards = []

        for step in range(args.steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward

            step_queues.append(info.get('total_queue_length', 0))
            step_waits.append(info.get('avg_waiting_time', 0))
            step_throughputs.append(info.get('step_throughput', 0))
            step_delays.append(info.get('avg_delay', 0))
            step_rewards.append(reward)

            if done:
                break

        # Episode summary
        ep_result = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'avg_queue_length': float(np.mean(step_queues)),
            'max_queue_length': float(np.max(step_queues)) if step_queues else 0,
            'avg_waiting_time': float(np.mean(step_waits)),
            'max_waiting_time': float(np.max(step_waits)) if step_waits else 0,
            'total_throughput': int(sum(step_throughputs)),
            'avg_delay': float(np.mean(step_delays)),
            'max_delay': float(np.max(step_delays)) if step_delays else 0,
            'step_queues': [float(x) for x in step_queues],
            'step_waits': [float(x) for x in step_waits],
            'step_throughputs': [int(x) for x in step_throughputs],
            'step_delays': [float(x) for x in step_delays],
            'step_rewards': [float(x) for x in step_rewards],
        }
        all_episode_results.append(ep_result)

        print(f"Episode {episode+1}/{args.episodes} | "
              f"Reward: {total_reward:10.2f} | "
              f"AvgQ: {ep_result['avg_queue_length']:6.1f} | "
              f"AvgWait: {ep_result['avg_waiting_time']:7.1f} | "
              f"Throughput: {ep_result['total_throughput']:4d} | "
              f"AvgDelay: {ep_result['avg_delay']:7.1f}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    env.close()

    avg_reward = np.mean([r['total_reward'] for r in all_episode_results])
    avg_queue = np.mean([r['avg_queue_length'] for r in all_episode_results])
    avg_wait = np.mean([r['avg_waiting_time'] for r in all_episode_results])
    avg_throughput = np.mean([r['total_throughput'] for r in all_episode_results])
    avg_delay = np.mean([r['avg_delay'] for r in all_episode_results])

    print(f"\n{'='*60}")
    print(f"Evaluation Summary — {agent_name}")
    print(f"{'='*60}")
    print(f"  Episodes:          {args.episodes}")
    print(f"  Avg Total Reward:  {avg_reward:.2f}")
    print(f"  Avg Queue Length:  {avg_queue:.2f}")
    print(f"  Avg Waiting Time:  {avg_wait:.2f} s")
    print(f"  Avg Throughput:    {avg_throughput:.1f} vehicles")
    print(f"  Avg Delay:         {avg_delay:.2f} s")

    # =========================================================================
    # Save Results
    # =========================================================================
    eval_data = {
        'agent_type': args.agent,
        'agent_name': agent_name,
        'num_episodes': args.episodes,
        'summary': {
            'avg_reward': avg_reward,
            'avg_queue_length': avg_queue,
            'avg_waiting_time': avg_wait,
            'avg_throughput': float(avg_throughput),
            'avg_delay': avg_delay,
        },
        'episodes': all_episode_results,
    }

    eval_path = os.path.join(results_dir, f"{args.agent}_eval.json")
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"\n[INFO] Evaluation results saved: {eval_path}")

    # =========================================================================
    # Generate per-agent evaluation plots
    # =========================================================================
    # Use the last episode for time-series plots
    last_ep = all_episode_results[-1]
    steps_range = range(1, len(last_ep['step_queues']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{agent_name} — Evaluation (Last Episode)', fontsize=16,
                 fontweight='bold')

    # Queue over time
    ax = axes[0, 0]
    ax.plot(steps_range, last_ep['step_queues'], color='#FF5722', linewidth=1.2)
    ax.set_title('Total Queue Length over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Length')
    ax.grid(True, alpha=0.3)

    # Waiting time over time
    ax = axes[0, 1]
    ax.plot(steps_range, last_ep['step_waits'], color='#3F51B5', linewidth=1.2)
    ax.set_title('Avg Waiting Time over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Wait Time (s)')
    ax.grid(True, alpha=0.3)

    # Cumulative throughput
    ax = axes[1, 0]
    cumulative_tp = np.cumsum(last_ep['step_throughputs'])
    ax.plot(steps_range, cumulative_tp, color='#4CAF50', linewidth=1.5)
    ax.set_title('Cumulative Throughput')
    ax.set_xlabel('Step')
    ax.set_ylabel('Vehicles')
    ax.grid(True, alpha=0.3)

    # Reward over time
    ax = axes[1, 1]
    ax.plot(steps_range, last_ep['step_rewards'], color='#FF9800', linewidth=1.0, alpha=0.7)
    cumulative_reward = np.cumsum(last_ep['step_rewards'])
    ax2 = ax.twinx()
    ax2.plot(steps_range, cumulative_reward, color='#E91E63', linewidth=1.5,
             label='Cumulative')
    ax.set_title('Reward over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Step Reward')
    ax2.set_ylabel('Cumulative Reward', color='#E91E63')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{args.agent}_eval.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Evaluation plot saved: {plot_path}")

    print("\n[SUCCESS] Evaluation completed!")
    print(f"\nTo compare all agents, run:")
    print(f"  python compare.py")


if __name__ == "__main__":
    main()

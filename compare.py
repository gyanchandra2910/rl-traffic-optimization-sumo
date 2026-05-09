"""
Cross-Algorithm Comparison Script

Loads evaluation results from all trained agents and generates
comparative visualizations and summary statistics.

Metrics compared:
    - Average Queue Length
    - Average Waiting Time
    - Throughput (vehicles/episode)
    - Average Delay
    - Total Reward

Usage:
    python compare.py
    python compare.py --results_dir results/
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv


# Agent display names and colors
AGENT_CONFIG = {
    'fixed_time':  {'name': 'Fixed-Time',       'color': '#9E9E9E', 'marker': 's'},
    'q_learning':  {'name': 'Q-Learning',        'color': '#2196F3', 'marker': 'o'},
    'double_q':    {'name': 'Double Q-Learning',  'color': '#FF9800', 'marker': '^'},
    'dqn':         {'name': 'DQN',                'color': '#4CAF50', 'marker': 'D'},
}

AGENT_ORDER = ['fixed_time', 'q_learning', 'double_q', 'dqn']


def load_eval_results(results_dir: str) -> dict:
    """
    Load evaluation JSON files for all available agents.

    Args:
        results_dir: Path to the results directory.

    Returns:
        dict: {agent_type: eval_data}
    """
    results = {}
    for agent_type in AGENT_ORDER:
        path = os.path.join(results_dir, f"{agent_type}_eval.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[agent_type] = json.load(f)
            print(f"  ✓ Loaded {AGENT_CONFIG[agent_type]['name']}")
        else:
            print(f"  ✗ Not found: {agent_type} (run evaluate.py --agent {agent_type} first)")
    return results


def load_training_results(results_dir: str) -> dict:
    """Load training JSON files for all available agents."""
    results = {}
    for agent_type in AGENT_ORDER:
        if agent_type == 'fixed_time':
            continue  # No training for fixed-time
        path = os.path.join(results_dir, f"{agent_type}_training.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[agent_type] = json.load(f)
    return results


def plot_training_convergence(training_data: dict, plots_dir: str):
    """Plot training convergence curves for all RL agents."""
    if not training_data:
        print("[SKIP] No training data found for convergence plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Convergence — All RL Agents', fontsize=16, fontweight='bold')

    metric_configs = [
        ('episode_rewards', 'Total Reward per Episode', 'Reward'),
        ('episode_avg_queues', 'Avg Queue Length per Episode', 'Queue Length'),
        ('episode_avg_waits', 'Avg Waiting Time per Episode', 'Wait Time (s)'),
        ('episode_throughputs', 'Throughput per Episode', 'Vehicles'),
    ]

    for idx, (key, title, ylabel) in enumerate(metric_configs):
        ax = axes[idx // 2, idx % 2]
        for agent_type, data in training_data.items():
            if key in data:
                cfg = AGENT_CONFIG[agent_type]
                values = data[key]
                episodes = range(1, len(values) + 1)
                ax.plot(episodes, values, color=cfg['color'], alpha=0.3, linewidth=0.8)
                # Add smoothed line
                if len(values) >= 10:
                    window = min(10, len(values))
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(range(window, len(values) + 1), smoothed,
                            color=cfg['color'], linewidth=2, label=cfg['name'])
                else:
                    ax.plot(episodes, values, color=cfg['color'], linewidth=2,
                            label=cfg['name'])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'training_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_metric_comparison_bars(eval_data: dict, plots_dir: str):
    """Generate bar chart comparisons for all key metrics."""
    if not eval_data:
        print("[SKIP] No evaluation data for bar charts.")
        return

    agents = [a for a in AGENT_ORDER if a in eval_data]
    names = [AGENT_CONFIG[a]['name'] for a in agents]
    colors = [AGENT_CONFIG[a]['color'] for a in agents]

    metrics = [
        ('avg_queue_length', 'Average Queue Length', 'Queue Length', True),
        ('avg_waiting_time', 'Average Waiting Time', 'Time (s)', True),
        ('avg_throughput', 'Average Throughput', 'Vehicles/Episode', False),
        ('avg_delay', 'Average Delay', 'Delay (s)', True),
        ('avg_reward', 'Average Total Reward', 'Reward', False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Algorithm Comparison — Key Metrics', fontsize=16, fontweight='bold')

    for idx, (metric_key, title, ylabel, lower_better) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = [eval_data[a]['summary'].get(metric_key, 0) for a in agents]
        bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='white',
                      linewidth=1.5)

        # Highlight best
        if lower_better:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        indicator = '(lower is better)' if lower_better else '(higher is better)'
        ax.set_xlabel(indicator, fontsize=8, color='gray')

    # Hide the extra subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    path = os.path.join(plots_dir, 'metric_comparison_bars.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_metric_boxplots(eval_data: dict, plots_dir: str):
    """Generate box plots for per-episode metric distributions."""
    if not eval_data:
        return

    agents = [a for a in AGENT_ORDER if a in eval_data]
    names = [AGENT_CONFIG[a]['name'] for a in agents]
    colors = [AGENT_CONFIG[a]['color'] for a in agents]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Metric Distribution Across Episodes', fontsize=14, fontweight='bold')

    metrics = [
        ('avg_queue_length', 'Avg Queue Length'),
        ('avg_waiting_time', 'Avg Waiting Time (s)'),
        ('total_throughput', 'Total Throughput'),
        ('avg_delay', 'Avg Delay (s)'),
    ]

    for idx, (key, ylabel) in enumerate(metrics):
        ax = axes[idx]
        data_lists = []
        for a in agents:
            ep_data = eval_data[a].get('episodes', [])
            vals = [ep.get(key, 0) for ep in ep_data]
            data_lists.append(vals)

        bp = ax.boxplot(data_lists, labels=names, patch_artist=True,
                        showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'red',
                                                    'markersize': 6})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'metric_boxplots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_time_series_comparison(eval_data: dict, plots_dir: str):
    """Plot time-series of metric evolution during an episode for all agents."""
    if not eval_data:
        return

    agents = [a for a in AGENT_ORDER if a in eval_data]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Time-Series Comparison (Last Evaluation Episode)',
                 fontsize=14, fontweight='bold')

    series_configs = [
        ('step_queues', 'Total Queue Length over Time', 'Queue Length'),
        ('step_waits', 'Avg Waiting Time over Time', 'Wait Time (s)'),
        ('step_delays', 'Avg Delay over Time', 'Delay (s)'),
        ('step_throughputs', 'Cumulative Throughput', 'Vehicles (cumulative)'),
    ]

    for idx, (key, title, ylabel) in enumerate(series_configs):
        ax = axes[idx // 2, idx % 2]
        for agent_type in agents:
            cfg = AGENT_CONFIG[agent_type]
            episodes = eval_data[agent_type].get('episodes', [])
            if not episodes:
                continue
            last_ep = episodes[-1]
            values = last_ep.get(key, [])
            if not values:
                continue

            if key == 'step_throughputs':
                values = np.cumsum(values).tolist()

            steps = range(1, len(values) + 1)
            ax.plot(steps, values, color=cfg['color'], linewidth=1.5,
                    label=cfg['name'], alpha=0.85)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'time_series_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def generate_summary_table(eval_data: dict, results_dir: str):
    """Generate and print a summary comparison table, and save to CSV."""
    if not eval_data:
        return

    agents = [a for a in AGENT_ORDER if a in eval_data]

    # Print table
    header = f"{'Agent':<22} {'Avg Queue':>10} {'Avg Wait(s)':>12} " \
             f"{'Throughput':>11} {'Avg Delay(s)':>13} {'Avg Reward':>11}"
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    rows = []
    for a in agents:
        s = eval_data[a]['summary']
        name = AGENT_CONFIG[a]['name']
        row = {
            'Agent': name,
            'Avg Queue Length': s.get('avg_queue_length', 0),
            'Avg Waiting Time (s)': s.get('avg_waiting_time', 0),
            'Throughput': s.get('avg_throughput', 0),
            'Avg Delay (s)': s.get('avg_delay', 0),
            'Avg Reward': s.get('avg_reward', 0),
        }
        rows.append(row)
        print(f"{name:<22} {row['Avg Queue Length']:>10.2f} "
              f"{row['Avg Waiting Time (s)']:>12.2f} "
              f"{row['Throughput']:>11.1f} "
              f"{row['Avg Delay (s)']:>13.2f} "
              f"{row['Avg Reward']:>11.2f}")

    print(separator)

    # Find best for each metric (lower is better for queue/wait/delay, higher for throughput/reward)
    if rows:
        best_queue = min(rows, key=lambda r: r['Avg Queue Length'])['Agent']
        best_wait = min(rows, key=lambda r: r['Avg Waiting Time (s)'])['Agent']
        best_tp = max(rows, key=lambda r: r['Throughput'])['Agent']
        best_delay = min(rows, key=lambda r: r['Avg Delay (s)'])['Agent']
        best_reward = max(rows, key=lambda r: r['Avg Reward'])['Agent']

        print(f"\n🏆 Best Queue: {best_queue}")
        print(f"🏆 Best Wait: {best_wait}")
        print(f"🏆 Best Throughput: {best_tp}")
        print(f"🏆 Best Delay: {best_delay}")
        print(f"🏆 Best Reward: {best_reward}")

    # Save to CSV
    csv_path = os.path.join(results_dir, 'comparison_summary.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Summary table saved: {csv_path}")


def plot_improvement_over_baseline(eval_data: dict, plots_dir: str):
    """Plot percentage improvement of RL agents over fixed-time baseline."""
    if 'fixed_time' not in eval_data:
        print("[SKIP] No fixed-time baseline data for improvement plot.")
        return

    baseline = eval_data['fixed_time']['summary']
    rl_agents = [a for a in AGENT_ORDER if a in eval_data and a != 'fixed_time']

    if not rl_agents:
        return

    metrics = [
        ('avg_queue_length', 'Queue Length', True),
        ('avg_waiting_time', 'Waiting Time', True),
        ('avg_throughput', 'Throughput', False),
        ('avg_delay', 'Delay', True),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Improvement Over Fixed-Time Baseline (%)', fontsize=14,
                 fontweight='bold')

    x = np.arange(len(metrics))
    width = 0.25
    n_agents = len(rl_agents)

    for i, agent_type in enumerate(rl_agents):
        cfg = AGENT_CONFIG[agent_type]
        improvements = []
        for metric_key, _, lower_better in metrics:
            base_val = baseline.get(metric_key, 1)
            agent_val = eval_data[agent_type]['summary'].get(metric_key, 0)
            if base_val != 0:
                if lower_better:
                    imp = ((base_val - agent_val) / abs(base_val)) * 100
                else:
                    imp = ((agent_val - base_val) / abs(base_val)) * 100
            else:
                imp = 0
            improvements.append(imp)

        offset = (i - n_agents/2 + 0.5) * width
        bars = ax.bar(x + offset, improvements, width, label=cfg['name'],
                      color=cfg['color'], alpha=0.8, edgecolor='white')
        for bar, imp_val in zip(bars, improvements):
            color = '#4CAF50' if imp_val > 0 else '#F44336'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{imp_val:+.1f}%', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel('Improvement (%)')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'improvement_over_baseline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def main():
    """Run the full comparison pipeline."""

    parser = argparse.ArgumentParser(description='Compare Traffic Light Agents')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(script_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("Cross-Algorithm Comparison")
    print("=" * 60)

    # Load data
    print("\n📂 Loading evaluation results...")
    eval_data = load_eval_results(results_dir)

    print("\n📂 Loading training results...")
    training_data = load_training_results(results_dir)

    if not eval_data:
        print("\n[ERROR] No evaluation results found!")
        print("Run evaluate.py for each agent first:")
        for a in AGENT_ORDER:
            print(f"  python evaluate.py --agent {a} --episodes 5")
        return

    # Generate all plots
    print("\n📊 Generating comparison plots...")
    plot_training_convergence(training_data, plots_dir)
    plot_metric_comparison_bars(eval_data, plots_dir)
    plot_metric_boxplots(eval_data, plots_dir)
    plot_time_series_comparison(eval_data, plots_dir)
    plot_improvement_over_baseline(eval_data, plots_dir)

    # Summary table
    print("\n📋 Comparison Summary:")
    generate_summary_table(eval_data, results_dir)

    print(f"\n{'='*60}")
    print("[SUCCESS] All comparison plots saved to:")
    print(f"  {plots_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

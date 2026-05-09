import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_plots():
    results_dir = 'results'
    agents = ['fixed_time', 'q_learning', 'double_q', 'dqn', 'a2c', 'ppo', 'trpo']
    
    agent_names = {
        'fixed_time': 'Fixed-Time Baseline',
        'q_learning': 'Q-Learning',
        'double_q': 'Double Q-Learning',
        'dqn': 'Deep Q-Network (DQN)',
        'a2c': 'Advantage Actor-Critic (A2C)',
        'ppo': 'Proximal Policy Optimization (PPO)',
        'trpo': 'Trust Region Policy Optimization (TRPO)'
    }
    
    colors = {
        'reward': '#2196F3', # Blue
        'queue': '#FF9800',  # Orange
        'wait': '#9C27B0',   # Purple
        'tp': '#4CAF50'      # Green
    }

    print("Generating 2x2 training plots for LaTeX report...")

    for agent in agents:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax_rew = axes[0, 0]
        ax_q = axes[0, 1]
        ax_wait = axes[1, 0]
        ax_tp = axes[1, 1]
        
        # Fixed Time Baseline doesn't train, so we extract its evaluation averages
        # and plot them as a flat baseline across 300 episodes for comparison.
        if agent == 'fixed_time':
            eval_path = os.path.join(results_dir, f"{agent}_eval.json")
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    data = json.load(f)
                
                avg_reward = data['summary']['avg_reward']
                avg_queue = data['summary']['avg_queue_length']
                avg_wait = data['summary']['avg_waiting_time']
                avg_tp = data['summary']['avg_throughput']
                
                episodes = range(1, 301)
                
                ax_rew.plot(episodes, [avg_reward]*300, color=colors['reward'], linewidth=2, label='Baseline')
                ax_q.plot(episodes, [avg_queue]*300, color=colors['queue'], linewidth=2, label='Baseline')
                ax_wait.plot(episodes, [avg_wait]*300, color=colors['wait'], linewidth=2, label='Baseline')
                ax_tp.plot(episodes, [avg_tp]*300, color=colors['tp'], linewidth=2, label='Baseline')
            else:
                print(f"[WARNING] Could not find {eval_path}")
                
        # RL Agents pull from their 300-episode training logs
        else:
            train_path = os.path.join(results_dir, f"{agent}_training.json")
            if os.path.exists(train_path):
                with open(train_path, 'r') as f:
                    data = json.load(f)
                    
                rewards = data.get('episode_rewards', [])
                queues = data.get('episode_avg_queues', [])
                waits = data.get('episode_avg_waits', [])
                tps = data.get('episode_throughputs', [])
                
                episodes = range(1, len(rewards) + 1)
                
                # Plot Raw Data
                ax_rew.plot(episodes, rewards, color=colors['reward'], alpha=0.3, label='Raw Data')
                ax_q.plot(episodes, queues, color=colors['queue'], alpha=0.3, label='Raw Data')
                ax_wait.plot(episodes, waits, color=colors['wait'], alpha=0.3, label='Raw Data')
                ax_tp.plot(episodes, tps, color=colors['tp'], alpha=0.3, label='Raw Data')
                
                # Add Moving Averages (Window = 20)
                window = min(20, len(rewards))
                if len(rewards) >= window:
                    smoothed_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    smoothed_q = np.convolve(queues, np.ones(window)/window, mode='valid')
                    smoothed_w = np.convolve(waits, np.ones(window)/window, mode='valid')
                    smoothed_tp = np.convolve(tps, np.ones(window)/window, mode='valid')
                    
                    x_sm = range(window, len(rewards)+1)
                    ax_rew.plot(x_sm, smoothed_r, color=colors['reward'], linewidth=2, label=f'MA ({window})')
                    ax_q.plot(x_sm, smoothed_q, color=colors['queue'], linewidth=2, label=f'MA ({window})')
                    ax_wait.plot(x_sm, smoothed_w, color=colors['wait'], linewidth=2, label=f'MA ({window})')
                    ax_tp.plot(x_sm, smoothed_tp, color=colors['tp'], linewidth=2, label=f'MA ({window})')
            else:
                print(f"[WARNING] Could not find {train_path}")
                
        # Formatting Plot 1: Rewards
        ax_rew.set_title('Total Reward vs. Episode', fontsize=12)
        ax_rew.set_xlabel('Episodes')
        ax_rew.set_ylabel('Total Reward')
        ax_rew.legend()
        ax_rew.grid(True, alpha=0.3)
        
        # Formatting Plot 2: Queues
        ax_q.set_title('Average Queue Length vs. Episode', fontsize=12)
        ax_q.set_xlabel('Episodes')
        ax_q.set_ylabel('Avg Queue Length (vehicles)')
        ax_q.legend()
        ax_q.grid(True, alpha=0.3)
        
        # Formatting Plot 3: Waits
        ax_wait.set_title('Average Wait Time vs. Episode', fontsize=12)
        ax_wait.set_xlabel('Episodes')
        ax_wait.set_ylabel('Avg Wait Time (s)')
        ax_wait.legend()
        ax_wait.grid(True, alpha=0.3)
        
        # Formatting Plot 4: Throughputs
        ax_tp.set_title('Total Vehicles (Throughput) vs. Episode', fontsize=12)
        ax_tp.set_xlabel('Episodes')
        ax_tp.set_ylabel('Throughput (vehicles)')
        ax_tp.legend()
        ax_tp.grid(True, alpha=0.3)
        
        # Main Title & Layout
        plt.suptitle(f'{agent_names[agent]}: Training Progression', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save exact filename required by LaTeX
        save_path = f"{agent}_training.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    print("\n[SUCCESS] All 2x2 plots generated perfectly for report.tex!")

if __name__ == '__main__':
    generate_plots()

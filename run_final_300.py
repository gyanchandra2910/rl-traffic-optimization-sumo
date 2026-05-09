"""
Main Training & Evaluation Pipeline for Traffic Signal Optimization.

Context in Project: "Traffic Signal Optimization using RL"
This script is the master entrypoint for the final 300-episode comparative run.
It iterates through all 7 algorithms (Fixed-Time, Q-Learning, Double Q, DQN,
A2C, PPO, TRPO), training and subsequently evaluating them in the SUMO
simulation environment, and finally triggers the comparison graph generators.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Environment
from env.sumo_env import SumoEnv

# Import Agents
from agent.fixed_time_agent import FixedTimeAgent
from agent.q_agent import QLearningAgent
from agent.double_q_agent import DoubleQLearningAgent
from agent.dqn_agent import DQNAgent
from agent.a2c_agent import A2CAgent
from agent.ppo_agent import PPOAgent
from agent.trpo_agent import TRPOAgent

# Import Compare module
import compare

# List of all algorithms evaluated in the thesis
AGENT_TYPES = ['fixed_time', 'q_learning', 'double_q', 'dqn', 'a2c', 'ppo', 'trpo']

# Meta-configuration for agents mapping them to their file extensions and plot aesthetics
AGENT_CONFIGS = {
    'fixed_time': {'name': 'Fixed-Time', 'ext': 'none', 'color': '#9E9E9E', 'marker': 's'},
    'q_learning': {'name': 'Q-Learning', 'ext': 'pkl', 'color': '#2196F3', 'marker': 'o'},
    'double_q':   {'name': 'Double Q', 'ext': 'pkl', 'color': '#FF9800', 'marker': '^'},
    'dqn':        {'name': 'DQN', 'ext': 'pt', 'color': '#4CAF50', 'marker': 'D'},
    'a2c':        {'name': 'A2C', 'ext': 'pt', 'color': '#E91E63', 'marker': 'v'},
    'ppo':        {'name': 'PPO', 'ext': 'pt', 'color': '#9C27B0', 'marker': '*'},
    'trpo':       {'name': 'TRPO', 'ext': 'pt', 'color': '#00BCD4', 'marker': 'p'}
}

def create_agent(agent_type: str, eval_mode: bool = False):
    """
    Instantiate the correct agent object based on the requested type.
    
    Why this matters:
    Ensures that during the 'Evaluation Phase', epsilon exploration is completely 
    turned off (epsilon=0.0) so the agent acts strictly deterministically.
    """
    epsilon = 0.0 if eval_mode else 1.0
    if agent_type == 'fixed_time':
        return FixedTimeAgent(action_dim=5, green_duration=30, delta_time=10)
    elif agent_type == 'q_learning':
        return QLearningAgent(action_dim=5, epsilon=epsilon)
    elif agent_type == 'double_q':
        return DoubleQLearningAgent(action_dim=5, epsilon=epsilon)
    elif agent_type == 'dqn':
        return DQNAgent(state_dim=6, action_dim=5, epsilon=epsilon)
    elif agent_type == 'a2c':
        return A2CAgent(state_dim=6, action_dim=5)
    elif agent_type == 'ppo':
        return PPOAgent(state_dim=6, action_dim=5)
    elif agent_type == 'trpo':
        return TRPOAgent(state_dim=6, action_dim=5)
    else:
        raise ValueError(f"Unknown agent: {agent_type}")


def print_header(title):
    """Utility function to print styled terminal headers."""
    print(f"\n{'='*80}")
    print(f"🌟 {title.center(76)}")
    print(f"{'='*80}")


def run_pipeline():
    """
    The master execution function.

    Architectural Flow:
    1. Initialize the SUMO environment safely.
    2. Loop through all 7 configured agents.
    3. TRAIN PHASE: Train the agent for 300 episodes, decay exploration, and save metrics.
    4. EVAL PHASE: Load the strictly trained deterministic model and test it for 3 episodes.
    5. Save all results strictly as JSON artifacts.
    6. Run the `compare.py` script to generate high-quality thesis visualizations.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(script_dir, "data", "network.net.xml")
    route_file = os.path.join(script_dir, "data", "routes.rou.xml")
    results_dir = os.path.join(script_dir, "results")
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print_header("Initializing SUMO Environment for Final 300-Episode Run")
    # Setting use_gui=False dramatically accelerates training speeds
    env = SumoEnv(net_file=net_file, route_file=route_file, use_gui=False, sim_max_steps=3600, delta_time=10)
    
    num_train_episodes = 300
    num_eval_episodes = 3
    max_steps = 360

    for agent_type in AGENT_TYPES:
        name = AGENT_CONFIGS[agent_type]['name']
        ext = AGENT_CONFIGS[agent_type]['ext']
        model_path = os.path.join(models_dir, f"{agent_type}_model.{ext}")
        
        print_header(f"Processing Agent: {name}")
        
        # ==========================================
        # 1. Training Phase
        # ==========================================
        if agent_type != 'fixed_time':
            print(f"|-- [TRAIN] Starting {num_train_episodes} episodes...")
            agent = create_agent(agent_type, eval_mode=False)
            
            episode_rewards = []
            episode_avg_queues = []
            episode_avg_waits = []
            episode_throughputs = []
            episode_avg_delays = []
            
            for episode in range(num_train_episodes):
                state = env.reset()
                total_reward = 0
                step_queues, step_waits, step_throughputs, step_delays = [], [], [], []
                
                for step in range(max_steps):
                    action = agent.choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    # Core RL update step
                    agent.learn(state, action, reward, next_state)
                    
                    state = next_state
                    total_reward += reward
                    step_queues.append(info.get('total_queue_length', 0))
                    step_waits.append(info.get('avg_waiting_time', 0))
                    step_throughputs.append(info.get('step_throughput', 0))
                    step_delays.append(info.get('avg_delay', 0))
                    
                    if done:
                        break
                        
                agent.decay_epsilon()
                
                episode_rewards.append(total_reward)
                episode_avg_queues.append(float(np.mean(step_queues) if step_queues else 0))
                episode_avg_waits.append(float(np.mean(step_waits) if step_waits else 0))
                episode_throughputs.append(int(sum(step_throughputs)))
                episode_avg_delays.append(float(np.mean(step_delays) if step_delays else 0))
                
                # Print progress every 20 episodes
                if (episode + 1) % 20 == 0 or episode == 0 or episode == num_train_episodes - 1:
                    print(f"|    Ep {episode+1:3d}/{num_train_episodes} -> Reward: {total_reward:8.1f} | Avg Queue: {episode_avg_queues[-1]:6.1f} | Throughput: {episode_throughputs[-1]:4d}")
                    
            # Persist learned intelligence to disk
            agent.save(model_path)
            print(f"|-- [TRAIN] Saved model to {model_path}")
            
            # Save Training Metrics as JSON for Plotting Generators
            train_metrics = {
                'agent_type': agent_type,
                'episodes': num_train_episodes,
                'episode_rewards': episode_rewards,
                'episode_avg_queues': episode_avg_queues,
                'episode_avg_waits': episode_avg_waits,
                'episode_throughputs': episode_throughputs,
                'episode_avg_delays': episode_avg_delays,
            }
            with open(os.path.join(results_dir, f"{agent_type}_training.json"), 'w') as f:
                json.dump(train_metrics, f)
        
        # ==========================================
        # 2. Evaluation Phase
        # ==========================================
        print(f"|-- [EVAL] Starting {num_eval_episodes} episodes...")
        eval_agent = create_agent(agent_type, eval_mode=True)
        
        # We strictly load the serialized model to ensure weights and normalization
        # running stats (Welford's variables) are identically maintained.
        if agent_type != 'fixed_time' and os.path.exists(model_path):
            eval_agent.load(model_path)
            
        all_eval_results = []
        for episode in range(num_eval_episodes):
            state = env.reset()
            total_reward = 0
            step_queues, step_waits, step_throughputs, step_delays, step_rewards = [], [], [], [], []
            
            for step in range(max_steps):
                action = eval_agent.choose_action(state)
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
                    
            ep_res = {
                'episode': episode + 1,
                'total_reward': float(total_reward),
                'avg_queue_length': float(np.mean(step_queues)),
                'avg_waiting_time': float(np.mean(step_waits)),
                'total_throughput': int(sum(step_throughputs)),
                'avg_delay': float(np.mean(step_delays)),
                'step_queues': [float(x) for x in step_queues],
                'step_waits': [float(x) for x in step_waits],
                'step_throughputs': [int(x) for x in step_throughputs],
                'step_delays': [float(x) for x in step_delays],
                'step_rewards': [float(x) for x in step_rewards],
            }
            all_eval_results.append(ep_res)
            print(f"|    Ep {episode+1}/{num_eval_episodes} -> Reward: {total_reward:8.1f} | Avg Queue: {ep_res['avg_queue_length']:6.1f} | Throughput: {ep_res['total_throughput']:4d}")

        # Save Evaluation Metrics
        eval_data = {
            'agent_type': agent_type,
            'agent_name': name,
            'num_episodes': num_eval_episodes,
            'summary': {
                'avg_reward': float(np.mean([r['total_reward'] for r in all_eval_results])),
                'avg_queue_length': float(np.mean([r['avg_queue_length'] for r in all_eval_results])),
                'avg_waiting_time': float(np.mean([r['avg_waiting_time'] for r in all_eval_results])),
                'avg_throughput': float(np.mean([r['total_throughput'] for r in all_eval_results])),
                'avg_delay': float(np.mean([r['avg_delay'] for r in all_eval_results])),
            },
            'episodes': all_eval_results,
        }
        with open(os.path.join(results_dir, f"{agent_type}_eval.json"), 'w') as f:
            json.dump(eval_data, f, indent=2)

    env.close()
    
    # ==========================================
    # 3. Graph Generation & Comparison
    # ==========================================
    print_header("Generating Final Metric Graphs")
    
    # Monkey-patch compare.py so it plots all 7 agents on one uniform scale
    compare.AGENT_ORDER = AGENT_TYPES
    for t in AGENT_TYPES:
        if t not in compare.AGENT_CONFIG:
            compare.AGENT_CONFIG[t] = AGENT_CONFIGS[t]

    sys.argv = ['compare.py', '--results_dir', results_dir]
    compare.main()
    
    print_header("300-Episode Pipeline Complete!")
    print(f"All 7 models saved in: {models_dir}")
    print(f"All final graphs saved in: {os.path.join(results_dir, 'plots')}")

if __name__ == "__main__":
    run_pipeline()

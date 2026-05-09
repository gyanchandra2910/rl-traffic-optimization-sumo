# Traffic Signal Optimization using Reinforcement Learning

## Overview

This project applies Reinforcement Learning algorithms to optimize traffic signal control at urban intersections. The goal is to minimize congestion, reduce waiting times, and maximize vehicle throughput by learning adaptive signal policies instead of using fixed time-based controllers.

The simulation environment is built using SUMO (Simulation of Urban Mobility), modeling a realistic Indian-style four-way intersection with left-hand drive traffic rules.

## Team Members

- Rohit Kumar (CS23B2053)
- Gyan Chandra (CS23I1053)
- Sarvan Kumar (ME23B1065)

Course Instructor: Dr. Rahul Raman

## Project Structure

```
├── agent/                 # RL agent implementations
│   ├── fixed_time_agent.py       # Baseline fixed-time controller
│   ├── q_agent.py                # Q-Learning
│   ├── double_q_agent.py         # Double Q-Learning
│   ├── dqn_agent.py              # Deep Q-Network
│   ├── a2c_agent.py              # Advantage Actor-Critic
│   ├── ppo_agent.py              # Proximal Policy Optimization
│   └── trpo_agent.py             # Trust Region Policy Optimization
├── env/                   # Environment wrapper
│   ├── sumo_env.py              # SUMO environment interface
│   └── traffic_mdp.py           # MDP specification
├── data/                  # SUMO configuration files
│   ├── network.net.xml
│   ├── routes.rou.xml
│   ├── trips.trips.xml
│   ├── sumo_config.sumocfg
│   ├── vtypes.add.xml
│   └── viewsettings.xml
├── models/                # Trained agent models
├── results/               # Evaluation results and plots
│   ├── comparison_summary.csv
│   ├── *_training.json
│   ├── *_eval.json
│   └── plots/
├── utils/                 # Utility functions
├── run_all.py             # Main pipeline: train all agents and generate comparisons
├── train.py               # Train individual agents
├── evaluate.py            # Evaluate trained agents
├── compare.py             # Generate comparison plots and statistics
├── ablation_runner.py     # Run reward function ablation study
├── report.tex             # Full technical report
└── README.md
```

## Control Strategies

### Fixed-Time (Baseline)
Non-learning baseline controller with static phase duration (30 seconds per phase).

### Tabular Methods
- **Q-Learning**: Value-based method with temporal-difference learning
- **Double Q-Learning**: Improved Q-Learning that addresses overestimation bias by using two value functions

### Deep Reinforcement Learning
- **DQN (Deep Q-Network)**: Neural network approximation with replay buffer and target network
- **A2C (Advantage Actor-Critic)**: Policy gradient method using actor-critic architecture
- **PPO (Proximal Policy Optimization)**: Safe policy gradient with clipped surrogate objective
- **TRPO (Trust Region Policy Optimization)**: Policy optimization with KL divergence constraint

## Key Results

From the evaluation over 300 episodes:

| Agent | Avg Queue | Wait (s) | Throughput | Delay (s) | Reward |
|-------|-----------|----------|-----------|-----------|--------|
| Fixed-Time | 14.22 | 6.79 | 188.0 | 8.92 | -1821.70 |
| Q-Learning | 11.71 | 4.47 | 191.0 | 6.72 | -1636.60 |
| **Double Q-Learning** | **11.11** | 4.57 | **200.0** | 6.81 | **-1427.50** |
| DQN | 12.45 | 5.08 | 190.7 | 7.29 | -1720.90 |
| A2C | 11.40 | 4.53 | 175.3 | 6.83 | -1540.70 |
| PPO | 12.04 | 5.05 | 184.0 | 7.30 | -1625.20 |
| **TRPO** | 11.18 | **4.26** | 182.0 | **6.55** | -1571.10 |

### Key Findings

1. **Double Q-Learning** achieves the best throughput (200 vehicles/episode)
2. **TRPO** achieves the best wait time (4.26s) and delay metrics (6.55s)
3. All RL methods outperform the fixed-time baseline
4. Value-based methods prioritize throughput; policy gradient methods prioritize wait time quality
5. Tabular methods remain highly competitive in this compact state space

## State Space

6-dimensional state vector:
- Queue length (North-South direction)
- Queue length (East-West direction)
- Waiting time aggregate (North-South direction)
- Waiting time aggregate (East-West direction)
- Active phase index
- Elapsed time in current phase

## Action Space

5 discrete actions:
- 0: Keep current phase
- 1-4: Switch to green phase 0-3

## Reward Function

```
R_t = -0.8 * ΔWait - 0.2 * Queue - 2.0 * SwitchCost + 1.0 * Throughput
```

This multi-objective reward was optimized through an ablation study on 10 different scenarios.

## Running the Project

### Prerequisites
- Python 3.8+
- SUMO (Simulation of Urban Mobility)
- PyTorch
- NumPy, Matplotlib

Set SUMO_HOME environment variable:
```bash
export SUMO_HOME=/path/to/sumo
```

### Run Full Pipeline
Train all agents and generate comparison plots:
```bash
python run_all.py
```

### Train Individual Agent
```bash
python train.py --agent q_learning --episodes 50
```

### Evaluate Agent
```bash
python evaluate.py --agent dqn --episodes 3
```

### Generate Comparison Plots
```bash
python compare.py --results_dir results/
```

## Output Files

- **models/**: Trained agent model files (.pkl for tabular, .pt for neural)
- **results/comparison_summary.csv**: Performance metrics table
- **results/*_training.json**: Training episode metrics
- **results/*_eval.json**: Evaluation episode details
- **results/plots/**: PNG visualizations
  - training_convergence.png
  - metric_comparison_bars.png
  - metric_boxplots.png
  - time_series_comparison.png
  - improvement_over_baseline.png

## Technical Notes

- State normalization (Welford's algorithm) is crucial for stable deep RL training
- Yellow phase duration is 3 seconds for realistic signal transitions
- Minimum green phase duration is 10 seconds
- Episode length is 360 steps (3600 simulation seconds with 10-second delta per step)

## References

See Traffic_Rl_Project(2).pdf for detailed mathematical formulations, MDP specification, and complete experimental methodology.

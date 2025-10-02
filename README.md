# Reinforcement Learning Projects Collection (will be uploaded soon, almost finished)

A collection of reinforcement learning implementations showcasing fundamental to advanced algorithms. Perfect for understanding RL concepts and their practical applications in autonomous systems and decision-making.

📋 Repository Structure

```
RL-Projects/
├── 1-Q-Learning-GridWorld/
│   ├── qlearning_gridworld.ipynb
│   ├── media/
│   │   ├── qgame_1.png
│   │   ├── qgame_2.png
│   │   └── qtable.png
│   └── README.md
├── 2-DeepQ-Network-CartPole/
│   ├── dqn_cartpole.ipynb
│   ├── cartpole_model/
│   │   └── DQN.keras
│   ├── media/
│   │   ├── cart_pole.gif
│   │   ├── DQN_policy_viz.png
│   │   └── dqn_net.png
│   └── README.md
├── 3-PPO-Pong/
│   ├── ppo_pong.ipynb
│   ├── ppo_models/
│   │   ├── pong_ppo_early.zip
│   │   └── pong_ppo_best.zip
│   ├── media/
│   │   ├── ponggif.gif
│   │   ├── ppo_model_arch.png
│   │   ├── action_prob_ppo.png
│   │   └── preprocessing.png
│   └── README.md
├── requirements.txt
└── README.md (main repository README)
```

# 📁 Projects Overview
## 1. 🤖 Q-Learning: Grid World Navigation
Screenshot: media/qgame_2.png

Key Features:
- Tabular Q-learning implementation
- Dynamic exploration-exploitation balance (ε-greedy)
- Optimal policy visualization
- Custom grid environment with obstacles

Results:
Best Actions Grid:
-------------------------------------------
| >:  94.06   | >:  96.02   | v:  98.00   | 
-------------------------------------------
| v:  96.02   |  OBSTACLE   | v: 100.00   | 
-------------------------------------------
| >:  98.00   | >: 100.00   |    GOAL     | 
-------------------------------------------

## 2. 🧠 Deep Q-Network: CartPole Balancing
Screenshot: media/DQN_policy_viz.png

Key Features:
- Neural network Q-value approximation
- Experience replay buffer
- Target network stabilization
- Double DQN implementation

Architecture:
Input (4) → Dense(24) → Dense(24) → Output(2)
Total Parameters: 770

Performance: Achieved perfect score of 500/500 consistently

## 3. 🎮 Proximal Policy Optimization: Atari Pong
Screenshot: media/action_prob_ppo.png

Key Features:
- Policy gradient optimization with clipping
- Convolutional neural network for pixel input
- Frame stacking and preprocessing
- Advantage estimation using GAE

Model Architecture:
- Input: 84x84x4 stacked frames
- CNN Backbone: 3 convolutional layers
- Heads: Actor (policy) + Critic (value)
- Training: 0M timesteps with stable learning

## 🛠️ Installation
```
git clone https://github.com/yourusername/RL-Projects.git
cd RL-Projects
```

## Install dependencies
```
pip install -r requirements.txt
```

requirements.txt:
```
gymnasium==0.29.1
tensorflow==2.15.0
stable-baselines3==2.0.0
numpy==1.24.3
ale-py==0.9.0
opencv-python==4.8.1
wandb==0.16.1
```


## 📊 Algorithm Comparison
Algorithm	Type	State Space	Action Space	Best For
Q-Learning	Value-based	Discrete	Discrete	Small, tabular environments
DQN	Value-based	Continuous	Discrete	High-dimensional observations
PPO	Policy-based	Continuous	Discrete/Continuous	Complex, stochastic environments

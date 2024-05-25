# Reinforcement Learning From Stratch
## Introduction
This repository contains the implementation of reinforcement learning algorithms from scratch. The purpose of this repository is to understand the reinforcement learning algorithms in depth. The repository is divided into the following topics:
 - Multi-Armed Bandit: This topic contains the implementation of epsilon-greedy, UCB, and Thompson samping algorithms. 
 - Dynamic Programming: Valued based approach, applicable for known environment.
   - value iteration: aims to find the optimal value function by iteratively updating the bellman value function. 
   - policy iteration: aims to find the optimal policy by iteratively updating the bellman Q equation.
 - Temporal Difference Learning: Applicable when the environment is unknown and the agent has to interact with the environment.
   - Monte Carlo Learning: Learning from complete episodes.
   - SARSA: On-policy TD control (learning optimal policy) algorithm.
   - Q-learning: Off-policy TD control algorithm.
 - Deep Q-Network: Deep reinforcement learning algorithm, apply deep learning to Q-learning.
   - DQN: Vanilla DQN.
   - Double DQN: the action used in the target was selected using a different network.
   - Dueling DQN: decompose the Q function into state value function and advantage function.
 - Policy Gradient: Policy-based reinforcement learning algorithm.
   - REINFORCE: Vanilla policy gradient algorithm, only one policy network, and was updated by the Monte Carlo return.
   - Actor-Critic: Policy-based reinforcement learning algorithm with value function. Actor generates the policy and critic evaluates the policy (estimates the V value).
   - PPO: Proximal Policy Optimization algorithm. Advanced TRPO algorithm, get rid of the KL penalty term by using the clipped surrogate objective. The actor network is optimized by the clipped ratio of the new policy and old policy. The critic network is optimized by the TD error.
   - DDPQ: distributed deep deterministic policy gradient algorithm. Actor-Critic algorithm with deterministic policy, able to handle continuous action space, off-policy learning.
   - SAC: Soft Actor-Critic algorithm. Actor-Critic algorithm with entropy regularization, learns a stochastic policy in an off-policy way.

## Setup
1. clone the repo into your local machine
```commandline
cd path/to/your/workspace
git clone https://github.com/Bingzw/reinforcement_learning_from_scratch.git
```
2. create a virtual environment
```commandline
python3 -m venv venv
source venv/bin/activate
```
3. install the required packages
```commandline
pip install -r requirements.txt
```
4. run the training script
```commandline
cd path/to/model_folder
python train.py
```

## Reference
- [EasyRL](https://github.com/datawhalechina/easy-rl)
- [Hands-On-RL](https://github.com/boyu-ai/Hands-on-RL)
- [Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)





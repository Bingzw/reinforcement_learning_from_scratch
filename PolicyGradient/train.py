import torch
import gymnasium as gym
import random
import numpy as np
from reinforce import REINFORCE
from util import plot_reward
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SEED = 0
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent_reinforce = REINFORCE(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                                learning_rate=learning_rate, gamma=gamma, device=device)
    agent_reinforce.train(num_episodes=num_episodes)
    plot_reward(reward_list=agent_reinforce.return_list, title="Reinforce")

    # play the game with the trained agent
    reinforce_reward = agent_reinforce.play(num_episodes=100)

    # plot the rewards in the same plot
    plt.plot(reinforce_reward, label='Vanilla DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of Vanilla DQN and Double DQN')
    plt.legend()
    plt.show()


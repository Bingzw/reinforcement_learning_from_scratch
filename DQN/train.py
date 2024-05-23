import torch
import gymnasium as gym
import random
import numpy as np
from dqn import DQN
from util import plot_reward
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SEED = 0
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    min_buffer_size = 500
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent_vanilla_dqn = DQN(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                            learning_rate=lr, gamma=gamma, epsilon=epsilon, target_update=target_update,
                            buffer_size=buffer_size, device=device, dqn_type='vanillaDQN')
    agent_double_dqn = DQN(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                            learning_rate=lr, gamma=gamma, epsilon=epsilon, target_update=target_update,
                            buffer_size=buffer_size, device=device, dqn_type='doubleDQN')

    agent_vanilla_dqn.train(num_episodes=num_episodes, batch_size=batch_size, min_buffer_size=min_buffer_size)
    plot_reward(reward_list=agent_vanilla_dqn.return_list, title="Vanilla DQN")

    agent_double_dqn.train(num_episodes=num_episodes, batch_size=batch_size, min_buffer_size=min_buffer_size)
    plot_reward(reward_list=agent_double_dqn.return_list, title="Double DQN")

    # plot two rewards in the same plot
    plt.plot(agent_vanilla_dqn.return_list, label='Vanilla DQN')
    plt.plot(agent_double_dqn.return_list, label='Double DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of Vanilla DQN and Double DQN')
    plt.legend()
    plt.show()







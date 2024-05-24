import torch
import gymnasium as gym
import random
import numpy as np
from reinforce import REINFORCE
from actor_critic import ActorCritic
from util import plot_reward
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SEED = 0
    reinforce_lr = 1e-3
    actor_lr = 1e-3
    critic_lr = 1e-2
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
                                learning_rate=reinforce_lr, gamma=gamma, device=device)
    agent_reinforce.train(num_episodes=num_episodes)
    plot_reward(reward_list=agent_reinforce.return_list, title="Reinforce")

    agent_actor_critic = ActorCritic(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                                        actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, device=device)
    agent_actor_critic.train(num_episodes=num_episodes)
    plot_reward(reward_list=agent_actor_critic.return_list, title="Actor Critic")

    # play the game with the trained agent
    reinforce_reward = agent_reinforce.play(num_episodes=100)
    actor_critic_reward = agent_actor_critic.play(num_episodes=100)

    # plot the rewards in the same plot
    plt.plot(reinforce_reward, label='Reinforce')
    plt.plot(actor_critic_reward, label='Actor Critic')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of Reinforce and Actor Critic')
    plt.legend()
    plt.show()


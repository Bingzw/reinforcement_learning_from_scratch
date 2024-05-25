import torch
import gymnasium as gym
import random
import numpy as np
from reinforce import REINFORCE
from actor_critic import ActorCritic
from ppo import PPO, PPOContinuous
from ddpq import DDPG, ReplayBuffer
from sac import SACContinuous
from util import plot_reward
import matplotlib.pyplot as plt


cartpole_params = {
    'env_name': 'CartPole-v1',
    'num_episodes': 1000,
    'hidden_dim': 128,
    'gamma': 0.98,
    # for REINFORCE
    'reinforce_lr': 1e-3,
    # for Actor-Critic and PPO
    'actor_lr': 1e-3,
    'critic_lr': 1e-2,
    # for PPO
    'lmbda': 0.9,
    'epochs': 10,
    'eps': 0.2
}

ppo_pendulum_params = {
    'env_name': 'Pendulum-v1',
    'num_episodes': 1000,
    'hidden_dim': 128,
    'gamma': 0.9,
    'actor_lr': 1e-4,
    'critic_lr': 5e-3,
    'lmbda': 0.9,
    'epochs': 10,
    'eps': 0.2
}

ddpg_pendulum_params = {
    'env_name': 'Pendulum-v1',
    'actor_lr': 3e-4,
    'critic_lr': 3e-3,
    'num_episodes': 200,
    'hidden_dim': 64,
    'gamma': 0.98,
    'tau': 0.005,
    'buffer_size': 10000,
    'minimal_size': 1000,
    'batch_size': 64,
    'sigma': 0.01

}

sac_pendulum_params = {
    'env_name': 'Pendulum-v1',
    'actor_lr': 3e-4,
    'critic_lr': 3e-3,
    'alpha_lr': 3e-4,
    'num_episodes': 100,
    'hidden_dim': 128,
    'gamma': 0.99,
    'tau': 0.005,
    'buffer_size': 10000,
    'minimal_size': 1000,
    'batch_size': 64,
}


if __name__ == '__main__':
    SEED = 0
    num_episodes = cartpole_params['num_episodes']
    hidden_dim = cartpole_params['hidden_dim']
    gamma = cartpole_params['gamma']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinforce_lr = cartpole_params['reinforce_lr']
    actor_lr = cartpole_params['actor_lr']
    critic_lr = cartpole_params['critic_lr']
    lmbda = cartpole_params['lmbda']
    epochs = cartpole_params['epochs']
    eps = cartpole_params['eps']

    # try different approaches on CartPole-v1
    env_name = cartpole_params['env_name']
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

    agent_ppo = PPO(env=env, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, actor_lr=actor_lr,
                    critic_lr=critic_lr, lmbda=lmbda, epochs=epochs, epsilon=eps, gamma=gamma, device=device)
    agent_ppo.train(num_episodes=num_episodes)
    plot_reward(reward_list=agent_ppo.return_list, title="PPO")

    # play the game with the trained agent
    reinforce_reward = agent_reinforce.play(num_episodes=100)
    actor_critic_reward = agent_actor_critic.play(num_episodes=100)
    ppo_reward = agent_ppo.play(num_episodes=100)

    # plot the rewards in the same plot
    plt.plot(reinforce_reward, label='Reinforce')
    plt.plot(actor_critic_reward, label='Actor Critic')
    plt.plot(ppo_reward, label='PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of Reinforce, Actor Critic and PPO')
    plt.legend()
    plt.show()

    # try different approaches on Pendulum-v1
    env_name = ppo_pendulum_params['env_name']
    env = gym.make(env_name)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent_ppo_c = PPOContinuous(env=env,
                                state_dim=state_dim,
                                hidden_dim=hidden_dim,
                                action_dim=action_dim,
                                actor_lr=ppo_pendulum_params['actor_lr'],
                                critic_lr=ppo_pendulum_params['critic_lr'],
                                lmbda=ppo_pendulum_params['lmbda'],
                                epochs=ppo_pendulum_params['epochs'],
                                epsilon=ppo_pendulum_params['eps'],
                                gamma=ppo_pendulum_params['gamma'],
                                device=device)
    agent_ppo_c.train(ppo_pendulum_params['num_episodes'])
    plot_reward(reward_list=agent_ppo_c.return_list, title="PPO_Continuous")

    ddpg_replay_buffer = ReplayBuffer(ddpg_pendulum_params['buffer_size'])
    agent_ddpg = DDPG(env=env,
                        state_dim=state_dim,
                        hidden_dim=hidden_dim,
                        action_dim=action_dim,
                        action_bound=action_bound,
                        actor_lr=ddpg_pendulum_params['actor_lr'],
                        critic_lr=ddpg_pendulum_params['critic_lr'],
                        gamma=ddpg_pendulum_params['gamma'],
                        tau=ddpg_pendulum_params['tau'],
                        sigma=ddpg_pendulum_params['sigma'],
                        device=device)
    agent_ddpg.train(ddpg_replay_buffer, ddpg_pendulum_params['num_episodes'], ddpg_pendulum_params['minimal_size'],
                     ddpg_pendulum_params['batch_size'])
    plot_reward(reward_list=agent_ddpg.return_list, title="DDPG")

    sac_replay_buffer = ReplayBuffer(sac_pendulum_params['buffer_size'])
    agent_sac = SACContinuous(env=env,
                              state_dim=state_dim,
                              hidden_dim=hidden_dim,
                              action_dim=action_dim,
                              action_bound=action_bound,
                              actor_lr=sac_pendulum_params['actor_lr'],
                              critic_lr=sac_pendulum_params['critic_lr'],
                              alpha_lr=sac_pendulum_params['alpha_lr'],
                              target_entropy=-env.action_space.shape[0],
                              gamma=sac_pendulum_params['gamma'],
                              tau=sac_pendulum_params['tau'],
                              device=device)
    agent_sac.train(sac_replay_buffer, sac_pendulum_params['num_episodes'], sac_pendulum_params['minimal_size'],
                    sac_pendulum_params['batch_size'])
    plot_reward(reward_list=agent_sac.return_list, title="SAC")

    # play the game with the trained agent
    ppo_c_reward = agent_ppo_c.play(num_episodes=100)
    ddpg_reward = agent_ddpg.play(num_episodes=100)
    sac_reward = agent_sac.play(num_episodes=100)
    # plot the rewards in the same plot
    plt.plot(ppo_c_reward, label='PPO_Continuous')
    plt.plot(ddpg_reward, label='DDPQ')
    plt.plot(sac_reward, label='SAC')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of PPO and DDPQ')
    plt.legend()
    plt.show()









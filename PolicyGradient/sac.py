import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()  # reparameterization trick, stochastic policy
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    """
    Soft Actor-Critic algorithm: an off-policy algorithm that learns a stochastic policy in an environment with
    continuous action space. It uses the maximum entropy reinforcement learning framework to maximize the expected
    return. It has three networks: actor, critic, and target critic. The actor network learns the policy, the critic
    network learns the Q value, and the target critic network is used to calculate the target value. It also has a
    temperature parameter alpha that is learned by gradient ascent. The entropy term is added to the Q value to
    encourage exploration. It also considers the minimum of the two Q values to stabilize training, which borrows the
    idea from the double Q learning.
    """
    def __init__(self, env, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr,
                 target_entropy, gamma, tau, device):
        self.env = env
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.return_list = []

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic1(next_states, next_actions)
        q2_value = self.target_critic2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * (1 - dones) * next_value
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0
        # update critic q networks
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic2(states, actions), td_target.detach()))
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        # update actor network
        actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic1(states, actions)
        q2_value = self.critic2(states, actions)
        q_value = torch.min(q1_value, q2_value)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def train(self, replay_buffer, num_episodes, minimal_size, batch_size):
        for i in range(num_episodes):
            episode_return = 0
            state, info = self.env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                action = self.take_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                        replay_buffer.sample(batch_size)
                    self.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            self.return_list.append(episode_return)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))

    def play(self, num_episodes):
        reward_list = []
        for i in range(num_episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0
            truncated = False
            while not done and not truncated:
                action = self.take_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                total_reward += reward
            reward_list.append(total_reward)
        return reward_list



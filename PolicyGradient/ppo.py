import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    """
    Policy network: output the probability distribution of actions given states, we are assuming discrete action space
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class ValueNet(nn.Module):
    """
    Value network: output the value of the state
    """
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PPO:
    """
    Proximal Policy Optimization algorithm: An advanced algorithm on top of TRPO. It uses a clipped objective function
    to prevent the policy from changing too much in each update.
    """
    def __init__(self, env, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, epsilon, gamma,
                 device):
        """
        :param env: environment
        :param state_dim: the dimension of the state space
        :param hidden_dim: the dimension of the hidden layer
        :param action_dim: the dimension of the action space
        :param actor_lr: actor learning rate
        :param critic_lr: critic learning rate
        :param lmbda: GAE parameter
        :param epochs: number of epochs
        :param epsilon: the range to clip the ratio
        :param gamma: discount factor
        :param device: device
        """
        self.env = env
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.return_list = []

    def take_action(self, state):
        """
        Take action based on the policy network
        :param state: the state
        :return: action
        """
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calculate_advantage(self, gamma, lmbda, td_delta):
        """
        Calculate the advantage
        :param gamma: discount factor
        :param lmbda: GAE parameter
        :param td_delta: temporal difference error
        :return: tensor of advantage
        """
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float32)

    def update(self, state_list, action_list, reward_list, next_state_list, done_list):
        """
        Update the actor and critic networks
        :param state_list: a list of states
        :param action_list: a list of actions
        :param reward_list: a list of rewards
        :param next_state_list: a list of next states
        :param done_list: a list of done
        """
        states = torch.tensor(state_list, dtype=torch.float32).to(self.device)
        actions = torch.tensor(action_list, dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(reward_list, dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_state_list, dtype=torch.float32).to(self.device)
        dones = torch.tensor(done_list, dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.calculate_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def train(self, num_episodes):
        for i in range(num_episodes):
            episode_reward = 0
            state, info = self.env.reset()
            state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
            done = False
            truncated = False
            while not done and not truncated:
                action = self.take_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_reward += reward
            self.return_list.append(episode_reward)
            self.update(state_list, action_list, reward_list, next_state_list, done_list)
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


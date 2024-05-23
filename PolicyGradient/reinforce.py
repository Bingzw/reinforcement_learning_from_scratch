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


class REINFORCE:
    """
    REINFORCE algorithm: Monte Carlo policy based method that directly optimizes the policy network. It is an
    on-policy method. In each episode, it samples a trajectory from the environment(Monte Carlo), and then updates
    the policy network.
    """
    def __init__(self, env, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        """
        :param env: environment
        :param state_dim: the dimension of the state space
        :param hidden_dim: the dimension of the hidden layer
        :param action_dim: the dimension of the action space
        :param learning_rate: learning rate
        :param gamma: discount factor
        :param device: device
        """
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.env = env
        self.gamma = gamma
        self.device = device
        self.return_list = []  # store the return for each episode

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, state_list, action_list, reward_list):
        """
        Update the policy network
        :param state_list: a list of states
        :param action_list: a list of actions
        :param reward_list: a list of rewards
        """
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float32).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        for i in range(num_episodes):
            episode_reward = 0
            state, info = self.env.reset()
            state_list, action_list, reward_list = [], [], []
            done = False
            truncated = False
            while not done and not truncated:
                action = self.take_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                state = next_state
                episode_reward += reward
            self.return_list.append(episode_reward)
            self.update(state_list, action_list, reward_list)
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



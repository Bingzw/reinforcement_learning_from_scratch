import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer:
    """
    Replay buffer to store past experiences that the agent can then use for training data.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        return np.array(batch_state), batch_action, batch_reward, np.array(batch_next_state), batch_done

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    """
    Deep Q Network: off-policy Q-learning with function approximation, usually used when the state space is large or
    continuous and the action space is discrete. It uses a neural network to approximate the Q function instead of doing
    table lookup.
    """
    def __init__(self, env, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, buffer_size,
                 device):
        """
        :param env: environment
        :param state_dim: the dimension of the state space
        :param hidden_dim: the hidden dimension of the Q network
        :param action_dim: the dimension of the action space
        :param learning_rate: learning rate for the optimizer
        :param gamma: discount factor in Q learning
        :param epsilon: probability of choosing a random action in the greedy policy
        :param target_update: the frequency of updating the target network
        :param buffer_size: the size of the replay buffer
        :param device: device to run the Q network on
        """
        self.env = env
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.return_list = []  # store the return for each episode

    def take_action(self, state):
        """
        greedy policy with epsilon probability to take random action
        :param state: the current state
        :return: action to take
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            action = torch.argmax(self.q_net(state)).item()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        """
        update Q values
        :param state: current state
        :param action: current action
        :param reward: reward
        :param next_state: next state
        :param done: whether the episode is over
        :return: None
        """
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones).to(torch.int32).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # get the q values of the actions taken
        max_next_q_values = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets.detach()))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # update target network
        self.count += 1

    def train(self, num_episodes, batch_size, min_buffer_size):
        for i in range(num_episodes):
            episode_reward = 0
            state, info = self.env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                action = self.take_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if self.replay_buffer.size() > min_buffer_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = self.replay_buffer.sample(batch_size)
                    self.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_done)
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))










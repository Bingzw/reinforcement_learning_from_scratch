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


class ActorCritic:
    """
    Actor-Critic algorithm: combines the advantages of both policy gradient and value-based methods. The actor network
    learns the policy, and the critic network learns the value function. When updating the critic network, we use the
    temporal difference error to update the value function. When updating the actor network, we followed the same target
    derived from policy gradient.
    """
    def __init__(self, env, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        """
        :param env: environment
        :param state_dim: the dimension of the state space
        :param hidden_dim: the dimension of the hidden layer
        :param action_dim: the dimension of the action space
        :param actor_lr: actor learning rate
        :param critic_lr: critic learning rate
        :param gamma: discount factor
        :param device: device
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.env = env
        self.gamma = gamma
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

        # the actor loss is derived from policy gradient.
        # 1. Initially, d_theta(J)/d_theta = E[Q(s,a)*d_theta(log(pi(a|s)))]
        # 2. the Q(s, a) can be estimated by value function V(s), Q(s, a) = E[r + gamma * V(s')]
        # 3. usually, we would like to add baseline function to reduce variance,
        # so the Q(s, a) = E[r + gamma * (V(s') - V(s))]
        # 4. In production, ignoring the expectation usually works better to allow more explorations, so
        # Q(s, a) = r + gamma * (V(s') - V(s))
        td_target = rewards + self.gamma * (1 - dones) * self.critic(next_states)  # note that unlike DQN which
        # maintains a target network, here we are using the same critic network to estimate the value of the next state
        # (without updating the parameters). The main reason that a target critic network is not needed is:
        # The q network in DQN is used to take the max action, so it is important to have a stable target network.
        # While the actor network in actor-critic is used to sample the action, so the stability is not as important as
        # DQN, it's updating parameters more frequently than DQN, thus also reducing the necessity of a target network.
        td_delta = td_target - self.critic(states)
        # actor network loss
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())  # adding detach to prevent backpropagation
        # the critic loss is the mse between critic predicted value and actural returns
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())  # adding detach to prevent backpropagation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
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







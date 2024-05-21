import numpy as np
from collections import defaultdict


class MonteCarlo:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4, num_episodes=500, seed=0):
        """
        :param env: environment
        :param gamma: discount factor
        :param alpha: learning rate
        :param epsilon: epsilon for epsilon-greedy policy
        :param n_actions: number of actions
        :param num_episodes: number of episodes
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((self.env.ncol * self.env.nrow, n_actions))
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.returns_sum = defaultdict(float)  # store the return for each state-action pair simulation
        self.returns_count = defaultdict(float)  # store the count of each state-action pair simulation
        self.return_list = []  # store the return for each episode
        self.seed = seed

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q[state])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):
            if self.Q[state][i] == Q_max:
                a[i] = 1
        return a

    def update(self, episode_trajectory):
        sa_in_episode = set([(x[0], x[1]) for x in episode_trajectory])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # find the first occurrence of the state-action pair in the episode
            first_occurrence_idx = next(i for i, x in enumerate(episode_trajectory) if x[0] == state and x[1] == action)
            # sum up all rewards since the first occurrence
            G = sum(x[2] * (self.gamma ** i) for i, x in enumerate(episode_trajectory[first_occurrence_idx:]))
            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            self.Q[state][action] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]

    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            state = self.env.reset()
            episode = []  # a list of (state, action, reward) simulation results
            episode_reward = 0
            while True:  # run monte carlo simulation until done
                action = self.take_action(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                self.update(episode)  # update Q value here to speed up the training process since W was updated
                # multiple times per episode
                state = next_state
                episode_reward += reward
                if done:
                    break
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))



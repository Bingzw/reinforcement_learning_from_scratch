import numpy as np


class QLearning:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4, num_episodes=500, seed=0):
        """
        :param env: environment
        :param gamma: discount factor
        :param alpha: learning rate
        :param epsilon: epsilon for epsilon-greedy policy
        :param n_actions: number of actions
        :param num_episodes: number of episodes
        :param seed: random seed
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((self.env.ncol * self.env.nrow, n_actions))
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.seed = seed
        self.return_list = []  # store the return for each episode

    def take_action(self, state):
        """
        greedy policy with epsilon probability to take random action
        :param state: the current state
        :return: action to take
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])  # only take the first best action when tie happens
        return action

    def best_action(self, state):
        """
        get the best action given state
        :param state: the current state
        :return: best actions
        """
        Q_max = np.max(self.Q[state])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):
            if self.Q[state][i] == Q_max:
                a[i] = 1
        return a

    def update(self, state, action, reward, next_state):
        """
        update Q values
        :param state: current state
        :param action: current action
        :param reward: reward
        :param next_state: next state
        :return: None
        """
        next_action = np.argmax(self.Q[next_state])  # off-policy, choose the best action no matter how the next
        # state is reached
        td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.take_action(state)  #
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                self.update(state, action, reward, next_state)
                state = next_state
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))
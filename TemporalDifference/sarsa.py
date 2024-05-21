import numpy as np


class SARSA:
    """
    On policy TD control
    """
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

    def update(self, state, action, reward, next_state, next_action):
        """
        update Q values
        :param state: current state
        :param action: current action
        :param reward: reward
        :param next_state: next state
        :param next_action: next action
        :return: None
        """
        td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            episode_reward = 0
            state = self.env.reset()
            action = self.take_action(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.take_action(next_state)  # on-policy, the policy used in this step denotes the
                # behavior policy (greedy policy in this case)
                episode_reward += reward
                self.update(state, action, reward, next_state, next_action)  # on-policy update, the target policy
                # (the actions used to update td error, which denotes the policy in the update function) is aligned
                # with (taken from the same policy) the behavior policy (the actions generated from the sampling)
                state = next_state
                action = next_action
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))


class NStepSARSA:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4, n_step=1, num_episodes=500, seed=0):
        """
        :param env: environment
        :param gamma: discount factor
        :param alpha: learning rate
        :param epsilon: epsilon for epsilon-greedy policy
        :param n_actions: number of actions
        :param n_step: n-step
        :param num_episodes: number of episodes
        :param seed: random seed
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((self.env.ncol * self.env.nrow, n_actions))
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.n = n_step
        self.state_list = []
        self.action_list = []
        self.reward_list = []
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
            action = np.argmax(self.Q[state])
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

    def update(self, state, action, reward, next_state, next_action, done):
        """
        update Q values
        :param state: current state
        :param action: current action
        :param reward: reward
        :param next_state: next state
        :param next_action: next action
        :param done: whether the episode is done
        :return: None
        """
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        if len(self.state_list) == self.n:
            G = self.Q[next_state][next_action]
            for i in reversed(range(self.n)):  # accumulate reward from the latest state, action
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:  # update the Q value even if there is less than n steps for the state, action
                    # (this is critical to make the algorithm work). For these states, we are updating the Q value like
                    # using Monte Carlo method
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q[s][a] += self.alpha * (G - self.Q[s][a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q[s][a] += self.alpha * (G - self.Q[s][a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []

    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            episode_reward = 0
            state = self.env.reset()
            action = self.take_action(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.take_action(next_state)
                episode_reward += reward
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))










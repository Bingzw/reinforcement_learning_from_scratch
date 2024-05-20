import numpy as np


class SARSA:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4):
        """
        :param env: environment
        :param gamma: discount factor
        :param alpha: learning rate
        :param epsilon: epsilon for epsilon-greedy policy
        :param n_actions: number of actions
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((self.env.ncol * self.env.nrow, n_actions))
        self.n_actions = n_actions
        self.epsilon = epsilon

    def take_action(self, state):
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

    def update(self, state, action, reward, next_state, next_action, **kwargs):
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


class NStepSARSA:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4, n_step=1):
        """
        :param env: environment
        :param gamma: discount factor
        :param alpha: learning rate
        :param epsilon: epsilon for epsilon-greedy policy
        :param n_actions: number of actions
        :param n_step: n-step
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

    def update(self, state, action, reward, next_state, next_action, **kwargs):
        done = kwargs.get("done", False)
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        if len(self.state_list) == self.n:
            G = self.Q[next_state][next_action]
            for i in reversed(range(self.n)):  # accumulate reward from the latest state, action
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:  # update the Q value even if there is less than n steps for the state, action
                    # (this is critical to make the algorithm work)
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






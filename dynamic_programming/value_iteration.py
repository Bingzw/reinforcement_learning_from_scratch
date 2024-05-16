class ValueIteration:
    def __init__(self, env, theta, gamma, n_actions=4):
        """
        :param env: environment
        :param theta: threshold for stopping iterations
        :param gamma: discount factor
        """
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.V = [0] * self.env.ncol * self.env.nrow
        self.policy = [None] * self.env.ncol * self.env.nrow
        self.n_actions = n_actions

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_V = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # list of state-action values, in the order of up, down, left, right
                for action_idx in range(self.n_actions):
                    qsa = 0
                    for p, next_state, reward, done in self.env.P[s][action_idx]:  # note that the reward is the reward
                        # given the state and action to next_state, we should multiple the transition probability p to
                        # get the expected reward given the state and action
                        qsa += p * (reward + self.gamma * self.V[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_V[s] = max(qsa_list)  # get the best state-action value
                max_diff = max(max_diff, abs(new_V[s] - self.V[s]))
            self.V = new_V
            cnt += 1
            if max_diff < self.theta:
                break
        print("Ran value iterations for {} times".format(cnt))
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []  # list of state-action values, in the order of up, down, left, right
            for action_idx in range(self.n_actions):
                qsa = 0
                for p, next_state, reward, done in self.env.P[s][action_idx]:
                    qsa += reward + self.gamma * p * self.V[next_state] * (1 - done)
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)
            self.policy[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]







import copy


class PolicyIteration:
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
        self.policy = [[0.25] * 4] * self.env.ncol * self.env.nrow  # uniform policy in the beginning
        self.n_actions = n_actions

    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_V = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for action_idx in range(self.n_actions):
                    qsa = 0
                    for p, next_state, reward, done in self.env.P[s][action_idx]:
                        qsa += p * (reward + self.gamma * self.V[next_state] * (1 - done))
                    qsa = qsa * self.policy[s][action_idx]
                    qsa_list.append(qsa)
                new_V[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_V[s] - self.V[s]))
            self.V = new_V
            if max_diff < self.theta:
                break
            cnt += 1
        print("Ran policy evaluation for {} times".format(cnt))

    def policy_improvement(self):
        # update policy based on the current value function, note that this is a greedy policy that has the same
        # implementation as the get_policy method in value_iteration.py
        for s in range(self.env.nrow * self.env.ncol):
            qsa_lsit = []
            for a in range(self.n_actions):
                qsa = 0
                for p, next_state, reward, done in self.env.P[s][a]:
                    qsa += p * (reward + self.gamma * self.V[next_state] * (1 - done))
                qsa_lsit.append(qsa)
            max_q = max(qsa_lsit)
            cnt_q = qsa_lsit.count(max_q)
            self.policy[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_lsit]
        return self.policy

    def policy_iteration(self):
        cnt = 1
        while True:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            new_policy = self.policy_improvement()
            if new_policy == old_policy:
                break
            self.policy = new_policy
            cnt += 1
        print("Ran policy iteration for {} times".format(cnt))




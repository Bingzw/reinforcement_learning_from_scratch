import numpy as np


class BaseSolver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # the number of times each arm has been pulled
        self.regret = 0.0  # cumulative regret
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # return the index of the arm to pull
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(BaseSolver):
    """
    Epsilon-Greedy algorithm: with probability epsilon, choose a random arm, otherwise choose the best arm
    """
    def __init__(self, bandit, epsilon, init_prob=1.0):
        """
        :param bandit: the multi-armed bandit environment
        :param epsilon: the probability of choosing a random arm
        :param init_prob: the initial expected reward of each arm
        """
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = init_prob * np.ones(self.bandit.K)  # initialize the expected rewards of each arm

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.choice(self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        reward = self.bandit.pull(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])  # update the expected reward estimate
        return k


class DecayEpsilonGreedy(EpsilonGreedy):
    """
    Decay Epsilon-Greedy algorithm: epsilon decreases over time
    """
    def __init__(self, bandit, init_epsilon, decay_rate, init_prob=1.0):
        super(DecayEpsilonGreedy, self).__init__(bandit, init_epsilon, init_prob)
        self.decay_rate = decay_rate

    def run_one_step(self):
        self.epsilon *= self.decay_rate
        return super(DecayEpsilonGreedy, self).run_one_step()


class UCB(BaseSolver):
    """
    Upper Confidence Bound (UCB) algorithm
    """
    def __init__(self, bandit, coef, init_prob=1.0):
        """
        :param bandit: the multi-armed bandit environment
        :param coef: the coef to control the weight of the upper confidence bound
        :param init_prob: the initial expected reward of each arm
        """
        super(UCB, self).__init__(bandit)
        self.coef = coef
        self.total_count = 0
        self.estimates = init_prob * np.ones(self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.pull(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class ThompsonSampling(BaseSolver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.alpha = np.ones(self.bandit.K)
        self.beta = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self.alpha, self.beta)
        k = np.argmax(samples)
        r = self.bandit.pull(k)
        self.alpha[k] += r
        self.beta[k] += 1 - r
        return k










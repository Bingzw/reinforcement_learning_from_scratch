import numpy as np


class MABEnv:
    def __init__(self, K, seed=100):
        """
        :param K: number of arms
        """
        np.random.seed(seed)
        self.probs = np.random.rand(K)  # initialize the probabilities of each arm
        self.best_idx = np.argmax(self.probs)  # the best arm
        self.K = K  # number of arms
        self.best_prob = self.probs[self.best_idx]  # the probability of the best arm

    def pull(self, k):
        """
        :param k: the index of the arm to pull
        :return: the reward of the arm, the reward is 1 with probability probs[k], otherwise 0
        """
        return np.random.binomial(1, self.probs[k])


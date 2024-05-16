class CliffWalkingWithPEnv:
    def __init__(self, ncol=12, nrow=4):
        self.nrow = nrow
        self.ncol = ncol
        self.action = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # up, down, left, right
        self.P = self.createP()  # [(p, next_state, reward, done)]

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]  # a list of list of tuples, each corresponds
        # the p, next state, reward, done given the current state and action
        for i in range(self.nrow):
            for j in range(self.ncol):
                for action_idx in range(len(self.action)):
                    cur_state = i * self.ncol + j
                    if i == self.nrow-1 and j > 0:
                        # reach to the cliff
                        next_state = i * self.ncol + j
                        reward = 0
                        done = True
                        P[cur_state][action_idx] = [(1.0, next_state, reward, done)]
                    else:
                        next_i = min(self.ncol-1, max(0, j + self.action[action_idx][0]))
                        next_j = min(self.nrow-1, max(0, i + self.action[action_idx][1]))
                        next_state = next_j * self.ncol + next_i
                        reward = -1
                        done = False
                        if next_j == self.nrow-1 and next_i > 0:
                            done = True
                            if next_i != self.ncol-1:  # not goal, on the cliff
                                reward = -100
                        P[cur_state][action_idx] = [(1.0, next_state, reward, done)]
        return P



if __name__ == "__main__":
    env = CliffWalkingWithPEnv()
    print(env.P)
    print(env.action)


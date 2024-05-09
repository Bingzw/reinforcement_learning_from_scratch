import matplotlib.pyplot as plt
from environment.mab_env import MABEnv
from MultiArmBandit.mab import EpsilonGreedy, DecayEpsilonGreedy, UCB, ThompsonSampling


def plot_results(solvers, solver_names):
    """

    :param solvers:
    :param solver_names:
    :return:
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # set up mab environment
    K = 10  # number of arms
    T = 5000  # number of time steps
    mab_env = MABEnv(K, seed=100)
    # epsilon greedy solvers
    epsilon_greedy_solvers = EpsilonGreedy(mab_env, epsilon=0.1)
    epsilon_greedy_solvers.run(T)
    # decay epsilon greedy solvers
    decay_epsilon_greedy_solvers = DecayEpsilonGreedy(mab_env, init_epsilon=0.1, decay_rate=0.99)
    decay_epsilon_greedy_solvers.run(T)
    # UCB solvers
    ucb_solvers = UCB(mab_env, coef=1)
    ucb_solvers.run(T)
    # Thompson Sampling solvers
    thompson_sampling_solvers = ThompsonSampling(mab_env)
    thompson_sampling_solvers.run(T)

    all_solvers = [epsilon_greedy_solvers, decay_epsilon_greedy_solvers, ucb_solvers, thompson_sampling_solvers]
    all_solver_names = ['Epsilon-Greedy', 'Decay Epsilon-Greedy', 'UCB', 'Thompson Sampling']
    plot_results(all_solvers, all_solver_names)




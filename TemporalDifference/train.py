from montecarlo import MonteCarlo
from sarsa import SARSA, NStepSARSA
from qlearning import QLearning
from environment.cliffwalk_env import CliffWalkingWithoutPEnv
from util import show_cliffwalking_result


if __name__ == '__main__':
    SEED = 0
    ncol = 12
    nrow = 4
    env = CliffWalkingWithoutPEnv(ncol=ncol, nrow=nrow)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 500
    agent_mc = MonteCarlo(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4, num_episodes=num_episodes,
                          seed=SEED)
    agent_sarsa = SARSA(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4, num_episodes=num_episodes,
                        seed=SEED)
    agent_nsarsa = NStepSARSA(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4, n_step=5,
                              num_episodes=num_episodes, seed=SEED)
    agent_q = QLearning(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4, num_episodes=num_episodes,
                          seed=SEED)

    agent_mc.train()
    show_cliffwalking_result(agent=agent_mc, env=env, plot_title="Monte Carlo")

    agent_sarsa.train()
    show_cliffwalking_result(agent=agent_sarsa, env=env, plot_title="SARSA")

    agent_nsarsa.train()
    show_cliffwalking_result(agent=agent_nsarsa, env=env, plot_title="NStepSARSA")

    agent_q.train()
    show_cliffwalking_result(agent=agent_q, env=env, plot_title="QLearning")









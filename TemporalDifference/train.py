import numpy as np
from sarsa import SARSA, NStepSARSA
from environment.cliffwalk_env import CliffWalkingWithoutPEnv
from util import print_agent, plot_reward


def model_free_td_training(env, agent, num_episodes, plot_title="", SEED=42):
    np.random.seed(SEED)
    return_list = []
    for i in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.take_action(next_state)
            episode_reward += reward
            kwargs = {"done": done}
            agent.update(state, action, reward, next_state, next_action, **kwargs)
            state = next_state
            action = next_action
        return_list.append(episode_reward)
        if (i + 1) % 10 == 0:
            print("Average reward for the last 10 episodes with episodes {} to {} is: {}".format(i - 9, i + 1, np.mean(
                return_list[-10:])))

    plot_reward(return_list, title=plot_title)
    print(" ")
    print("Visualize the policy")
    action_meaning = ['^', 'v', '<', '>']
    print_agent(agent=agent, env=env, action_meaning=action_meaning, disaster=list(range(37, 47)), end=[47])


if __name__ == '__main__':
    SEED = 0
    ncol = 12
    nrow = 4
    env = CliffWalkingWithoutPEnv(ncol=ncol, nrow=nrow)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent_sarsa = SARSA(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4)
    agent_nsarsa = NStepSARSA(env=env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_actions=4, n_step=3)
    num_episodes = 500

    model_free_td_training(env, agent_sarsa, num_episodes, plot_title="SARSA")
    model_free_td_training(env, agent_nsarsa, num_episodes, plot_title="NStepSARSA")







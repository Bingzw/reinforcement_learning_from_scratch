import gymnasium as gym
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from environment.cliffwalk_env import CliffWalkingWithPEnv


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("value function：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.V[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("policy：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # print * for cliff positions and E for end position
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.policy[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    print("Cliff Walking Environment")
    cliff_walk_env = CliffWalkingWithPEnv()
    action_meaning_cw = ['^', 'v', '<', '>']
    theta_cw = 1e-6
    gamma_cw = 0.9
    n_actions_cw = 4
    vi_cw = ValueIteration(env=cliff_walk_env, theta=theta_cw, gamma=gamma_cw, n_actions=n_actions_cw)
    vi_cw.value_iteration()
    print("Cliff Walking Value Iteration Result:")
    print_agent(vi_cw, action_meaning_cw, list(range(37, 47)), [47])
    print("----------------")
    pi_cw = PolicyIteration(env=cliff_walk_env, theta=theta_cw, gamma=gamma_cw, n_actions=n_actions_cw)
    pi_cw.policy_iteration()
    print("Cliff Walking Policy Iteration Result:")
    print_agent(pi_cw, action_meaning_cw, list(range(37, 47)), [47])

    print("###########################")

    print("Frozen Lake Environment")
    frozen_lake_env = gym.make("FrozenLake-v1", render_mode='ansi')
    frozen_lake_env = frozen_lake_env.unwrapped
    frozen_lake_env.reset()
    print(frozen_lake_env.render())

    holes = set()
    ends = set()
    for s in frozen_lake_env.P:
        for a in frozen_lake_env.P[s]:
            for s_ in frozen_lake_env.P[s][a]:
                if s_[2] == 1.0:  # reward = 1.0
                    ends.add(s_[1])
                if s_[3] == True:
                    holes.add(s_[1])
    holes = holes - ends
    print("frozen index:", holes)
    print("goal index:", ends)

    n_actions_fl = frozen_lake_env.action_space.n
    theta_fl = 1e-5
    gamma_fl = 0.9
    vi_fl = ValueIteration(env=frozen_lake_env, theta=theta_fl, gamma=gamma_fl, n_actions=n_actions_fl)
    vi_fl.value_iteration()
    action_meaning_fl = ['<', 'v', '>', '^']
    print("Frozen Lake Value Iteration Result:")
    print_agent(vi_fl, action_meaning_fl, [5, 7, 11, 12], [15])
    print("----------------")
    pi_fl = PolicyIteration(env=frozen_lake_env, theta=theta_fl, gamma=gamma_fl, n_actions=n_actions_fl)
    pi_fl.policy_iteration()
    print("Frozen Lake Policy Iteration Result:")
    print_agent(pi_fl, action_meaning_fl, [5, 7, 11, 12], [15])








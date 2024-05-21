import matplotlib.pyplot as plt


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


def plot_reward(reward_list, title):
    plt.plot(reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()


def show_cliffwalking_result(agent, env, plot_title=""):
    plot_reward(agent.return_list, title=plot_title)
    print(" ")
    print("Visualize the policy")
    action_meaning = ['^', 'v', '<', '>']
    print_agent(agent=agent, env=env, action_meaning=action_meaning, disaster=list(range(37, 47)), end=[47])


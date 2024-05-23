import matplotlib.pyplot as plt


def plot_reward(reward_list, title):
    plt.plot(reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()
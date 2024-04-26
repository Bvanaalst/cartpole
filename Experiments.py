
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from Actor_with_entropy import reinforce
from Actor_critic_2 import actor_critic
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(agent, n_repetitions=20, n_episodes=501, learning_rate=1e-4,
                             gamma=0.99, smoothing_window=9, n=5):
    returns_over_repetitions = []
    #now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        if agent == 'actor':
            returns, timesteps, avg_rewards = reinforce(n_episodes, learning_rate, gamma)
        elif agent == 'actor_critic_bootstrap':
            baseline = False
            bootstrap = True
            returns, timesteps, avg_rewards = actor_critic(n_episodes, learning_rate, gamma, n, baseline, bootstrap)
        elif agent == 'actor_critic_baseline':
            baseline = True
            bootstrap = False
            returns, timesteps, avg_rewards = actor_critic(n_episodes, learning_rate, gamma, n, baseline, bootstrap)
        elif agent == 'actor_critic_bootstrap_baseline':
            baseline = True
            bootstrap = True
            returns, timesteps, avg_rewards = actor_critic(n_episodes, learning_rate, gamma, n, baseline, bootstrap)

        returns_over_repetitions.append(returns)

    #print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)  # average over repetitions
    if smoothing_window is not None:
        learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve, timesteps, avg_rewards

def experiment():
    ####### Settings
    n_repetitions = 1
    smoothing_window = 3
    n_episodes = 10
    gamma = 0.99
    learning_rate = 0.0001
    n = 5

    ####### Experiments

    Plot = LearningCurvePlot(title='Model Comparison: REINFORCE vs Actor-critic with bootstrapping and baseline subtraction')
    Plot.set_ylim(-100, 100)
    agent = 'actor'
    learning_curve, timesteps, avg_rewards = average_over_repetitions(agent, n_repetitions, n_episodes, learning_rate, gamma,
                                                         smoothing_window, n)
    Plot.add_curve(timesteps, learning_curve, label='REINFORCE')
    Plot.add_curve(timesteps, avg_rewards, label='REINFORCE-avg')

    agent = 'actor_critic_bootstrap'
    learning_curve, timesteps, avg_rewards = average_over_repetitions(agent, n_repetitions, n_episodes, learning_rate, gamma,
                                                         smoothing_window, n)
    Plot.add_curve(timesteps, learning_curve, label='Actor-critic-bootstrap')
    Plot.add_curve(timesteps, avg_rewards, label='Actor-critic-bootstrap-avg')

    agent = 'actor_critic_baseline'
    learning_curve, timesteps, avg_rewards = average_over_repetitions(agent, n_repetitions, n_episodes, learning_rate, gamma,
                                                         smoothing_window, n)
    Plot.add_curve(timesteps, learning_curve, label='Actor-critic-baseline')
    Plot.add_curve(timesteps, avg_rewards, label='Actor-critic-baseline-avg')

    agent = 'actor_critic_bootstrap_baseline'
    learning_curve, timesteps, avg_rewards = average_over_repetitions(agent, n_repetitions, n_episodes, learning_rate, gamma,
                                                         smoothing_window, n)
    Plot.add_curve(timesteps, learning_curve, label='Actor-critic-bootstrap-baseline')
    Plot.add_curve(timesteps, avg_rewards, label='Actor-critic-bootstrap-baseline-avg')

    Plot.save('actor_with_entropy.png')


if __name__ == '__main__':
    experiment()

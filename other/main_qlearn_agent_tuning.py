from environment import Environment
from agents import QLearningAgent
from simulator import Simulator
import pandas as pd
import numpy as np

stats = []

n_trials = 100
threshold = n_trials / 10

def aggregated_stats_build_row(q_init_value, alpha_rate, epsilon_rate, gamma_rate, stats_by_iteration):
    stats_sliced = stats_by_iteration[n_trials-threshold:]

    iterations_count = 0
    success_count = 0
    traffic_violations_count = 0
    explored_states_cum = 0
    reward_count = 0
    actions_count = 0

    for iteration_stats in stats_sliced:
        iterations_count += 1
        if iteration_stats['success']:
            success_count += 1
        traffic_violations_count = traffic_violations_count + iteration_stats['traffic_violations_count']
        explored_states_cum = explored_states_cum + iteration_stats['explored_states_cum']
        reward_count = reward_count + iteration_stats['cum_reward']
        actions_count = actions_count + iteration_stats['actions_count']

    row = {
        'q_init_value': q_init_value,
        'alpha_rate': alpha_rate,
        'epsilon_rate': epsilon_rate,
        'gamma_rate': gamma_rate,
        'success_perc': (float(success_count) / float(iterations_count)) * 100,
        'traffic_violations_avg': float(traffic_violations_count) / float(iterations_count),
        'explored_states_avg': float(explored_states_cum) / float(iterations_count),
        'reward_cum_avg': float(reward_count) / float(iterations_count),
        'actions_avg': float(actions_count) / float(iterations_count),
    }

    stats.append(row)


q_init_values = [0.0, 5.0, 10]
samples_to_generate = 3

for q_init_value in q_init_values:
    for alpha_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
        for epsilon_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
            for gamma_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
                print "Simulating for q_init_value: {}, alpha_rate: {}, epsilon_rate: {}, gamma_rate: {}". \
                    format(q_init_value, alpha_rate, epsilon_rate, gamma_rate)
                e = Environment()
                a = QLearningAgent(
                    e,
                    alpha_rate=alpha_rate, epsilon_rate=epsilon_rate, gamma_rate=gamma_rate, q_init_value=q_init_value
                )
                e.set_primary_agent(a, enforce_deadline=True)
                s = Simulator(e, update_delay=0.0000001, display=False)
                s.run(n_trials=n_trials)

                stats_by_iteration = a.stats_by_simulation_get()
                aggregated_stats_build_row(q_init_value, alpha_rate, epsilon_rate, gamma_rate, stats_by_iteration)

df = pd.DataFrame(
    data=stats, columns=['q_init_value', 'alpha_rate', 'epsilon_rate', 'gamma_rate', 'success_perc',
                         'traffic_violations_avg', 'explored_states_avg', 'reward_cum_avg', 'actions_avg']
)

df.to_csv('qlearn_agent_tuning_results_3_samples.csv')

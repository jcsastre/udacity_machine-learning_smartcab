from environment import Environment
from agents import QLearningAgent
from simulator import Simulator
import pandas as pd

stats = []

n_trials = 500
threshold = 20

def aggregated_stats_build_row(p_q_init_value, p_alpha_rate, p_epsilon_rate, p_gamma_rate, p_stats_by_round):
    stats_sliced = p_stats_by_round[n_trials - threshold:]

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
        'q_init_value': p_q_init_value,
        'alpha_rate': p_alpha_rate,
        'epsilon_rate': p_epsilon_rate,
        'gamma_rate': p_gamma_rate,
        'success_perc': (float(success_count) / float(iterations_count)) * 100,
        'traffic_violations_avg': float(traffic_violations_count) / float(iterations_count),
        'explored_states_avg': float(explored_states_cum) / float(iterations_count),
        'reward_cum_avg': float(reward_count) / float(iterations_count),
        'actions_avg': float(actions_count) / float(iterations_count),
    }

    stats.append(row)

for q_init_value in [0.0]:
    for alpha_rate in [0.5]:
        for epsilon_rate in [0.0]:
            for gamma_rate in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
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

df.to_csv('qlearn_agent_tuned_grid_for_gamma.csv')

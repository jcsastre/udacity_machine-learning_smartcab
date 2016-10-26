from environment import Environment
from agents import QLearningAgent
from simulator import Simulator
import pandas as pd
import numpy as np

stats = []

q_init_values = [0.0, 2.0, 4.0, 8.0, 16.0]

samples_to_generate = 21
n_trials = 100

for q_init_value in q_init_values:
    for alpha_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
        for epsilon_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
            for gamma_rate in np.linspace(0.00, 1.00, num=samples_to_generate):
                print "Simulating for q_init_value: {}, alpha_rate: {}, epsilon_rate: {}, gamma_rate: {}" . \
                    format(q_init_value, alpha_rate, epsilon_rate, gamma_rate)
                e = Environment()
                a = QLearningAgent(
                    e,
                    alpha_rate=alpha_rate, epsilon_rate=epsilon_rate, gamma_rate=gamma_rate, q_init_value=q_init_value
                )
                e.set_primary_agent(a, enforce_deadline=True)
                s = Simulator(e, update_delay=0.0000001, display=False)
                s.run(n_trials=n_trials)

                iteration_stats_aggregated = a.stats_get_aggregated()

                success_rate = \
                    float(iteration_stats_aggregated['successCount']) / float(iteration_stats_aggregated['iterationsCount'])
                success_perc = int(success_rate * 100)

                row = {
                    'alpha_rate': alpha_rate,
                    'epsilon_rate': epsilon_rate,
                    'gamma_rate': gamma_rate,
                    # 'successRate': str(iteration_stats_aggregated['successCount']) + "/" + str(
                    #     iteration_stats_aggregated['iterationsCount']),
                    'successPerc': success_perc,
                    'actionsAvg': iteration_stats_aggregated['actionsAvg'],
                    'cumRewardAvg': iteration_stats_aggregated['cumRewardAvg']
                    # 'iterationsCount': iteration_stats_aggregated['iterationsCount'],
                    # 'successCount': iteration_stats_aggregated['successCount'],
                }

                stats.append(row)

df = pd.DataFrame(
    data=stats, columns=['alpha_rate', 'epsilon_rate', 'gamma_rate', 'successPerc', 'actionsAvg', 'cumRewardAvg']
)

df.to_csv('qlearn_agent_tuning_results.csv')

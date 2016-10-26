from simulator import Simulator
from environment import Environment
from agents import QLearningAgent

environment = Environment(debug_traces=False)
qlearn_agent = QLearningAgent(environment)
environment.set_primary_agent(qlearn_agent, enforce_deadline=True)

simulator = Simulator(environment, update_delay=0.00001, display=False)
simulator.run(n_trials=100)

df = qlearn_agent.stats_by_simulation_get_as_df()
df.to_csv('stats_first_qlearn_agent.csv')

from simulator import Simulator
from environment import Environment
from agents import QLearningAgent

environment = Environment(debug_traces=True)

qlearn_agent_tuned = QLearningAgent(
    environment,
    alpha_rate=0.5,
    epsilon_rate=0.0,
    gamma_rate=0.5,
    q_init_value=0.0,
    debug_traces=True
)

environment.set_primary_agent(qlearn_agent_tuned, enforce_deadline=True)

simulator = Simulator(environment, update_delay=0.00001, display=False, debug_traces=True)
simulator.run(n_trials=100)

df = qlearn_agent_tuned.stats_by_simulation_get_as_df()
df.to_csv('stats_tuned_qlearn_agent.csv')

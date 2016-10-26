from smartcab.simulator import Simulator
from smartcab.environment import Environment
from smartcab.agents import QLearningAgent

environment = Environment()
qlearn_agent = QLearningAgent(environment)
environment.set_primary_agent(qlearn_agent, enforce_deadline=True)

simulator = Simulator(environment, update_delay=0.001, display=False)
simulator.run(n_trials=100)

print "******"
print "******"
print qlearn_agent.stats_iteration_get_as_df()

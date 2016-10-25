from smartcab.simulator import Simulator
from smartcab.environment import Environment
from smartcab.agents import RandomAgent

environment = Environment()
random_agent = RandomAgent(environment)
environment.set_primary_agent(random_agent, enforce_deadline=False)

simulator = Simulator(environment, update_delay=0.5, display=True)
simulator.run(n_trials=100)

from simulator import Simulator
from environment import Environment
from agents import RandomAgent

environment = Environment()
random_agent = RandomAgent(environment)
environment.set_primary_agent(random_agent, enforce_deadline=False)

simulator = Simulator(environment, update_delay=0.5, display=True)
simulator.run(n_trials=100)

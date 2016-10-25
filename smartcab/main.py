from environment import Environment
from agents import RandomAgent, QLearningAgent
from simulator import Simulator

# Set up environment and agent
e = Environment()  # create environment (also adds some dummy traffic)
a = e.create_agent(QLearningAgent)  # create agent
e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
# NOTE: You can set enforce_deadline=False while debugging to allow longer trials

# Now simulate it
sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True,
# if available)
# NOTE: To speed up simulation, reduce update_delay and/or set display=False

sim.run(n_trials=1)  # run for a specified number of trials
# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
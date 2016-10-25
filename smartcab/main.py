from environment import Environment
from agents import RandomAgent, QLearningAgent
from simulator import Simulator

# # Set up environment and agent
# e = Environment()  # create environment (also adds some dummy traffic)
# a = e.create_agent(QLearningAgent)  # create agent
# e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
# # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
#
# # Now simulate it
# sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True,
# # if available)
# # NOTE: To speed up simulation, reduce update_delay and/or set display=False
#
# sim.run(n_trials=1)  # run for a specified number of trials
# # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

import numpy as np

for alpha_rate in np.linspace(0.00, 1.00, num=2):
    for epsilon_rate in np.linspace(0.00, 1.00, num=2):
        for gamma_rate in np.linspace(0.00, 1.00, num=2):
            print "Simulating for alpha_rate: {}, epsilon_rate: {}, gamma_rate: {}" . format(alpha_rate, epsilon_rate, gamma_rate)
            e = Environment()
            a = QLearningAgent(e, alpha_rate=alpha_rate, epsilon_rate=epsilon_rate, gamma_rate=gamma_rate)
            e.set_primary_agent(a, enforce_deadline=True)
            s = Simulator(e, update_delay=0.001, display=False)
            s.run(n_trials=10)
            a.stats_print()
            print
            print

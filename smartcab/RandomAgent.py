import Agent
import pandas as pd
from RoutePlanner import RoutePlanner

class RandomAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(RandomAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.cum_reward = 0
        self.stats = []

    def stats_add_row(self, success):
        iteration = len(self.stats) + 1
        self.stats.append(
            (iteration, self.cum_reward, success)
        )

    def stats_print(self):
        df = pd.DataFrame(data=self.stats, columns=['iteration', 'cum_reward', 'success'])
        print df

    def stats_plot(self):
        df = pd.DataFrame(data=self.stats, columns=['iteration', 'cum_reward', 'success'])
        print len(df[df['success'] == True])
        print len(df[df['success'] == False])

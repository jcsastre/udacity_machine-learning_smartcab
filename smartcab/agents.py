import random
import pandas as pd

from routeplanner import RoutePlanner

# from altair import Chart
# import matplotlib.pyplot as plt

class Agent(object):
    """Base class for all agents."""

    def __init__(self, env):
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.color = 'cyan'

    def reset(self, destination=None):
        pass

    def get_cum_reward(self):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state

    def get_next_waypoint(self):
        return self.next_waypoint

class DummyAgent(Agent):
    color_choices = ['blue', 'cyan', 'magenta', 'orange']

    def __init__(self, env):
        super(DummyAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.next_waypoint = random.choice([None, 'forward', 'left', 'right'])
        self.color = random.choice(self.color_choices)

    def update(self, t):
        inputs = self.env.sense(self)

        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            self.next_waypoint = random.choice([None, 'forward', 'left', 'right'])
        reward = self.env.act(self, action)
        #print "DummyAgent.update(): t = {}, inputs = {}, action = {}, reward = {}".format(t, inputs, action, reward)  # [debug]
        #print "DummyAgent.update(): next_waypoint = {}".format(self.next_waypoint)  # [debug]

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

    def reset(self, destination=None):
        self.planner.route_to(destination)


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Random update of the state
        action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}" . \
            format(deadline, inputs, action, reward)

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha_rate=0.7, epsilon_rate=0.9, gamma_rate=0.5, q_init_value=10.0, debug_traces=False):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(QLearningAgent, self).__init__(env)

        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.valid_actions = [None, 'forward', 'left', 'right']

        self.alpha = alpha_rate  # Learning rate
        self.epsilon = epsilon_rate  # Exploration rate
        self.gamma = gamma_rate  # Discount factor rate
        self.q_init_value = q_init_value  # Initial value for the q matrix

        self.q_matrix = {}

        self.previous_state = None
        self.previous_action = None

        self.cum_reward = 0
        self.stats = []
        self.stats_by_simulation = []

        self.actions_count = 0
        self.traffic_violations_count = 0

        self.debug_traces = debug_traces

    def stats_by_simulation_add_row(self, success):
        row = {
            'simulation_round': len(self.stats_by_simulation) + 1,
            'success': success,
            'cum_reward': self.cum_reward,
            'explored_states_cum': len(self.q_matrix),
            'traffic_violations_count': self.traffic_violations_count,
            'actions_count': self.actions_count - 1,
        }

        self.stats_by_simulation.append(row)

    def stats_by_simulation_get(self):
        return self.stats_by_simulation

    def stats_by_simulation_get_as_df(self):
        df = pd.DataFrame(
            data=self.stats_by_simulation,
            columns=[
                'simulation_round',
                'success',
                'cum_reward',
                'explored_states_cum',
                'traffic_violations_count',
                'actions_count'
            ]
        )

        return df

    def stats_add_row(self, success):
        iteration = len(self.stats) + 1
        self.stats.append(
            (iteration, len(self.q_matrix), self.cum_reward, success, self.actions_count, self.moves_available)
        )

    def reset(self, destination=None):
        self.planner.route_to(destination)

        self.state = None
        self.previous_state = None
        self.previous_action = None

        self.cum_reward = 0

        self.actions_count = 0
        self.traffic_violations_count = 0

    def get_cum_reward(self):
        return self.cum_reward

    def get_q_value(self, state, action):
        key = (state, action)
        return self.q_matrix.get(key, self.q_init_value)

    def choose_action(self, state):
        if random.random() < self.epsilon:  # explore
            action = random.choice(self.valid_actions)
        else:  # exploit
            q = [self.get_q_value(state, a) for a in self.valid_actions]
            max_q = max(q)
            count = q.count(max_q)
            if count > 1:
                best = [i for i in range(len(self.valid_actions)) if q[i] == max_q]
                i = random.choice(best)
                action = self.valid_actions[i]
            else:
                i = q.index(max_q)
                action = self.valid_actions[i]

        return action

    def learn(self, previous_state, previous_action, reward, state):
        # Please, notice that get_q_value returns 0.0 is no Q-value found, this is equivalent to set initial condition
        # for Q-matrix
        old_q_value = self.get_q_value(previous_state, previous_action)

        max_q_new = max([self.get_q_value(state, a) for a in self.valid_actions])

        learned_value = reward + self.gamma * max_q_new

        new_q_value = old_q_value + self.alpha * (learned_value - old_q_value)

        self.q_matrix[(previous_state, previous_action)] = new_q_value

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = inputs
        self.state['next_waypoint'] = self.next_waypoint
        self.state = tuple(sorted(self.state.items()))

        # Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Code for stats purpose - BEGIN
        self.actions_count += 1
        if reward == -1.0:
            self.traffic_violations_count += 1
        self.cum_reward = self.cum_reward + reward
        # Code for stats purpose - END

        # Learn policy based on state, action, reward
        if reward is not None:
            self.learn(self.previous_state, self.previous_action, reward, self.state)

        # Set previous_state as current value of state
        self.previous_state = self.state
        self.previous_action = action

        if self.debug_traces:
            print "\tLearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}" . \
                format(deadline, inputs, action, reward)  # [debug]

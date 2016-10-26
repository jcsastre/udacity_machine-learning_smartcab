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

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
        #                                                                                             action, reward)  # [debug]

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha_rate=0.9, epsilon_rate=0.1, gamma_rate=0.5, q_init_value=0.0):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(QLearningAgent, self).__init__(env)

        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.valid_actions = [None, 'forward', 'left', 'right']

        self.alpha = alpha_rate  # Learning rate
        self.epsilon = epsilon_rate  # Exploration rate
        self.gamma = gamma_rate # Discount factor rate
        self.q_init_value = q_init_value  # Initial value for the q matrix

        self.q_matrix = {}

        self.previous_state = None
        self.previous_action = None

        self.cum_reward = 0
        self.stats = []

        self.actions_count = 0

    def stats_get_aggregated(self):
        iterations_count = 0
        success_count = 0
        actions_count = 0
        cum_reward_total = 0

        for row in self.stats:
            iterations_count += 1
            if row[3]:
                success_count += 1
            actions_count = actions_count + row[4]
            cum_reward_total = cum_reward_total + row[2]

        avg_actions = actions_count / iterations_count
        avg_cum_reward = cum_reward_total / iterations_count

        stats = {
            'iterationsCount': iterations_count,
            'successCount': success_count,
            'actionsAvg': avg_actions,
            'cumRewardAvg': avg_cum_reward
        }

        return stats

    def stats_add_row(self, success):
        iteration = len(self.stats) + 1
        self.stats.append(
            (iteration, len(self.q_matrix), self.cum_reward, success, self.actions_count)
        )

    # def stats_plot(self):
    #     df = pd.DataFrame(data=self.stats, columns=['iteration', 'q_size', 'cum_reward', 'success'])
    #     print len(df[df['success'] == True])
    #     print len(df[df['success'] == False])
    #
    #     # x = df['iteration']
    #     # y = df['q_size']
    #     # plt.scatter(x, y)
    #     # plt.xlim([0, max(x)+10])
    #     # plt.ylim([0, max(y)+10])
    #     # plt.xlabel("Number of iterations")
    #     # plt.ylabel("Q size")
    #     # plt.show()
    #     #
    #     # x = df['iteration']
    #     # y = df['cum_reward']
    #     # plt.scatter(x, y)
    #     # plt.xlim([0, max(x)+10])
    #     # plt.ylim([min(y)-10, max(y)+10])
    #     # plt.xlabel("Number of iterations")
    #     # plt.ylabel("Accumulated reward")
    #     # plt.show()
    #
    #     # foo = df.ix[df['success']==True]
    #     #
    #     # print df['success' == True]

    def reset(self, destination=None):
        self.planner.route_to(destination)

        self.state = None
        self.previous_state = None
        self.previous_action = None

        self.cum_reward = 0

        self.actions_count = 0

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

    # Old version of learn method
    # def learn(self, previous_state, previous_action, reward, state):
    #     old_q_value = self.q_matrix.get((previous_state, previous_action), None)
    #     if old_q_value is None:
    #         self.q_matrix[(previous_state, previous_action)] = reward
    #     else:
    #         max_q_new = max([self.get_q_value(state, a) for a in self.valid_actions])
    #         learned_value = reward + self.gamma * max_q_new
    #         self.q_matrix[(previous_state, previous_action)] = old_q_value + self.alpha * (learned_value - old_q_value)

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
        self.cum_reward = self.cum_reward + reward
        self.actions_count += 1

        # Learn policy based on state, action, reward
        if reward is not None:
            self.learn(self.previous_state, self.previous_action, reward, self.state)

        # Set previous_state as current value of state
        self.previous_state = self.state
        self.previous_action = action

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, q_size = {}".format(
        #     deadline, inputs, action, reward, len(self.q_matrix))  # [debug]

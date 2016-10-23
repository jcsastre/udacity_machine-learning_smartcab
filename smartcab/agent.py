# coding=utf-8
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# class RandomAgent(Agent):
#     """An agent that learns to drive in the smartcab world."""
#
#     def __init__(self, env):
#         super(RandomAgent, self).__init__(
#             env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
#         self.color = 'red'  # override color
#         self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
#
#     def reset(self, destination=None):
#         self.planner.route_to(destination)
#
#     def update(self, t):
#         # Gather inputs
#         self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
#         inputs = self.env.sense(self)
#         deadline = self.env.get_deadline(self)
#
#         # Random update of the state
#         action = random.choice([None, 'forward', 'left', 'right'])
#
#         # Execute action and get reward
#         reward = self.env.act(self, action)
#
#         print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon_rate=0.5, alpha_rate=0.5, gamma_rate=0.5):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(QLearningAgent, self).__init__(env)

        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.valid_actions = Environment.valid_actions

        self.epsilon = epsilon_rate  # Exploration rate
        self.alpha = alpha_rate  # Learning rate
        self.gamma = gamma_rate # Discount factor rate

        self.q_matrix = {}

        self.previous_state = None
        self.previous_action = None

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def get_q_value(self, state, action):
        key = (state, action)
        return self.q_matrix.get(key, 0.0)

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
        old_q_value = self.q_matrix.get((previous_state, previous_action), None)
        if old_q_value is None:
            self.q_matrix[(previous_state, previous_action)] = reward
        else:
            max_q_new = max([self.get_q_value(state, a) for a in self.valid_actions])
            learned_value = reward + self.gamma * max_q_new
            self.q_matrix[(previous_state, previous_action)] = old_q_value + self.alpha * (learned_value - old_q_value)

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

        # TODO: Learn policy based on state, action, reward
        if reward is not None:
            self.learn(self.previous_state, self.previous_action, reward, self.state)

        # Set previous_state as current value of state
        self.previous_state = self.state
        self.previous_action = action

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

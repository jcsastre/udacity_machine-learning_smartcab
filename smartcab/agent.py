import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha_rate=0.7, epsilon_rate=0.9, gamma_rate=0.5, q_init_value=10.0, debug_traces=False):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_actions = [None, 'forward', 'left', 'right']

        self.alpha = alpha_rate  # Learning rate
        self.epsilon = epsilon_rate  # Exploration rate
        self.gamma = gamma_rate  # Discount factor rate
        self.q_init_value = q_init_value  # Initial value for the q matrix
        self.debug_traces = debug_traces

        self.q_matrix = {}

        self.previous_state = None
        self.previous_action = None

        # self.cum_reward = 0
        # self.stats = []
        # self.stats_by_simulation = []
        #
        # self.actions_count = 0
        # self.traffic_violations_count = 0

    # def stats_by_simulation_add_row(self, success):
    #     row = {
    #         'simulation_round': len(self.stats_by_simulation) + 1,
    #         'success': success,
    #         'cum_reward': self.cum_reward,
    #         'explored_states_cum': len(self.q_matrix),
    #         'traffic_violations_count': self.traffic_violations_count,
    #         'actions_count': self.actions_count - 1,
    #     }
    #
    #     self.stats_by_simulation.append(row)
    #
    # def stats_by_simulation_get(self):
    #     return self.stats_by_simulation
    #
    # def stats_by_simulation_get_as_df(self):
    #     df = pd.DataFrame(
    #         data=self.stats_by_simulation,
    #         columns=[
    #             'simulation_round',
    #             'success',
    #             'cum_reward',
    #             'explored_states_cum',
    #             'traffic_violations_count',
    #             'actions_count'
    #         ]
    #     )
    #
    #     return df
    #
    # def stats_add_row(self, success):
    #     iteration = len(self.stats) + 1
    #     self.stats.append(
    #         (iteration, len(self.q_matrix), self.cum_reward, success, self.actions_count, self.moves_available)
    #     )

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.previous_state = None
        self.previous_action = None

        # self.cum_reward = 0
        #
        # self.actions_count = 0
        # self.traffic_violations_count = 0

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
        # Please, notice that get_q_value returns self.q_init_value (defined on constructor), if
        # no Q-value is found. This is equivalent to set initial condition for Q-matrix
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

        # TODO: Update state
        self.state = inputs
        self.state['next_waypoint'] = self.next_waypoint
        self.state = tuple(sorted(self.state.items()))

        # TODO: Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if reward is not None:
            self.learn(self.previous_state, self.previous_action, reward, self.state)

        # Set previous_state as current value of state
        self.previous_state = self.state
        self.previous_action = action

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}" . \
            format(deadline, inputs, action, reward)  # [debug]

        # # Code for stats purpose - BEGIN
        # self.actions_count += 1
        # if reward == -1.0:
        #     self.traffic_violations_count += 1
        # self.cum_reward = self.cum_reward + reward
        # # Code for stats purpose - END

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()

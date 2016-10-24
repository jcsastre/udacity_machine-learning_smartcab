<h1>Self-Driving Agent Report</h1>

<h2>Implementation of a Basic Driving Agent</h2>

As starting task, we will move the smartcab around the environment using 
a random approach. The set of possible actions will be: None, forward, 
left, right. The deadline will be set to false, but this doesn't mean
that smartcat has an infinite number of moves as can see on code of the file
"environment.py" (any way, enforcing it to false will increase a lot the
number of moves available).

The code corresponding to this agent can be found on the class 
"RandomAgent" at the file "agent.py".

<!--```-->
<!--action = random.choice([None, 'forward', 'left', 'right'])-->
<!--```-->

Observations from simulation:

1. Normally the smartcab action is not optimal, but normally reaches the
destination because has a lot of moves available to reach the destination.
2. Sometimes the smartcab doesn't reachs the destination (very few cases).
3. The environment  doesn't allow any agent to execute and action that
violates traffic rules, but a strong negative reward is applied.

<h2>Inform the Driving Agent</h2>

The next task  is to identify a set of states that are appropriate for modeling 
the smartcab and environment. 

All the information we receive come from the environment and the planner.

Sensing the environment provide us with these inputs:

- **light**
    - Possible values: Red / Green
- **oncoming**:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car oncoming and the action wants to execute.
- **right**:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the right oncoming and 
    the action wants to execute.
- **left**:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the left oncoming and 
    the action wants to execute.

Also from the environment we can obtain the **deadline** that is number of remaining
moves to reach the destination.

The planner provides **next_waypoint** with these possible values: Forward, Right
and Left.

For representing the state we will use: **next_waypoint**, **light**, **oncoming**, 
**right** and **left**.

Having in mind we have **next_waypoint**, is not very useful to use also **deadline**. Also
**deadline** will increase considerably the number of possible states, and would
penalize the Q-Learning implementation.

The information from **light**, **oncoming**, **right** and **left** can help Q-Learning
to avoid traffic violations. The information from **next_waypoint** can help Q-Learning
to reach the destination as soon as possible.
 
Having in mind the properties used for the state, and possible values for each of
these, the total number of different states are: 3 x 2 x 4 x 4 x 4. This means a
total of 384 states at a given time.

<h2>Implement a Q-Learning Driving Agent</h2>

The third task is to implement the Q-Learning algorithm for the driving agent. The
code corresponding to this agent can be found on the class *QLearningAgent* at the
file *agent.py*.

Before proceeding to the simulation, the values for three important constants should
be assigned:
- *alpha_rate (α)*: The learning rate. Determines to what extent the newly acquired 
information will override the old information.
- *epsilon_rate (ε)*: The exploration rate. Determines when to explore or
when to exploit the already learn information.
- *gamma rate (γ)*: The discount factor. Determines the importance of future 
rewards.

We will execute 100 simulations with enforce_deadline to True. We will do our first
attempt with these values for the constants:
- *alpha_rate (α)* = 0.9
- *epsilon_rate (ε)* = 0.1
- *gamma rate (γ)* = 0.5.

Let's analyze a scatter plot that correlates number of simulations executed (iterations) with
size of the Q matrix:

![](plots/qlearner_1_scatter_iterations_q-size.png)

As we can see in the plot, as simulations are executed the number of values in Q
matrix increases. At the beginning increases fast, but later increases slow. This
is normal because the number of scenarios not visited decreases while simulations
accumulate.

Let's analyze a scatter plot that correlates number of iterations with accumulated 
reward for each of the iterations:

![](plots/qlearner_1_scatter_iterations_cum-reward.png)

We don't see that as the number of iterations increase the agent gets better
accumulated rewards. Maybe the assigned values for the constants were poorly chosen.

Now let's see the number of times q-learner agent has achieved the destination:
- Success: 21 times
- Fail: 79 times

In the first section of this report we saw that the random agent normally was
successfully, but we must consider that the deadline was set to false. To make
a fairer comparison, lets compare with a random agent with deadline set to true.

Let's see the number of times random agent has achieved the destination:
- Success: 16 times
- Fail: 84 times

There is not too much difference between the success ratio of the *RandomAgent* and
the *QLearningAgent*. As said previously maybe the values for the constants were 
poorly chosen. Other options are that 100 simulations are not enough for the *QLearningAgent*,
or maybe the q-learn algorithm is bad implemented.




 Observations:

1. We see that normally the smartcab doesn't reach the destination. This is
because the deadline is set to True and there too few movements available.
2. As the simulations are executed (until n=100), the the number of Q-values increases. At
the beginning fast, but later slow. This is normal because the number of scenarios
 not visited decreases while simulations accumulate.
3. Theorically, as the simulations execute, the QLearningAgent learns, so hopefully
the number of negative rewards should decrease with time.

<!--Observations from simulation:-->

<!--1. The smartcab always reaches the destination. The reasons are that now the-->
<!--smartcab tries to move to next waypoint suggested by planner, and also that-->
<!--has a lot of moves available.-->
<!--2. The smartcab never receives a negative reward because now information about-->
<!--light and other cars is taken into account before executing any action.-->



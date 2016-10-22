<h1>Self-Driving Agent Report</h1>

<h2>Implementation of a Basic Driving Agent</h2>

As starting task, we will move the smartcab around the environment using 
a random approach. The set of possible actions will be: None, forward, 
left, right. The deadline will be set to false, but this doesn't mean
have an infinite number of moves as can see on code of the file
"environment.py" (any way, enforcing it to false will increase a lot the
number of moves available).

The code corresponding to this agent can be found on the class 
"RandomAgent" at the file "agent.py".

Observations:

1. Normally the smartcab action is not optimal, but normally reaches the
destination because has a lot of moves available to reach the destination.
2. Sometimes the smartcab doesn't reachs the destination (very few cases).
3. The environment  doesn't allow any agent to execute and action that
violates traffic rules, but a strong negative reward is applied.

<h2>Inform the Driving Agent</h2>

The next task is to identify a set of states useful to determine actions that 
allows to arrive to the next waypoint provided by the planner, but also these
actions have to obbey the traffic rules.

The possible values for next waypoint are: Forward / Right / Left
 
Each time agent senses the environment it receives these inputs:

- Light
    - Values: Red / Green
- Oncoming:
    - Values: None / Forward / Right / Left
    - Indicates if there is a car oncoming and the action wants to execute.
- Right:
    - Values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the right oncoming and 
    the action wants to execute.
- Left:
    - Values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the left oncoming and 
    the action wants to execute.


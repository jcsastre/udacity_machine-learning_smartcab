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

Observations from simulation:

1. Normally the smartcab action is not optimal, but normally reaches the
destination because has a lot of moves available to reach the destination.
2. Sometimes the smartcab doesn't reachs the destination (very few cases).
3. The environment  doesn't allow any agent to execute and action that
violates traffic rules, but a strong negative reward is applied.

<h2>Inform the Driving Agent</h2>

The next task  is to identify a set of states that are appropriate for modeling 
the smartcab and environment. 

The planner provides the property next_waypoint with these possible values: 
Forward, Right and Left.
 
Each time agent senses the environment it receives these properties:

- Light
    - Possible values: Red / Green
- Oncoming:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car oncoming and the action wants to execute.
- Right:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the right oncoming and 
    the action wants to execute.
- Left:
    - Possible values: None / Forward / Right / Left
    - Indicates if there is a car approaching from the left oncoming and 
    the action wants to execute.
    
All these properties will be used as state for the smartcab:

1. Next waypoint.
2. Light. 
3. Oncoming.
4. Right
5. Left
    
After each step we update the state, and then select an action that tries to
follow the Next Waypoint suggested by the planner, but without violating traffic
rules.
    
Observations from simulation:

1. The smartcab always reaches the destination. The reasons are that now the
smartcab tries to move to next waypoint suggested by planner, and also that
has a lot of moves available.
2. The smartcab never receives a negative reward because now information about
light and other cars is taken into account before executing any action.



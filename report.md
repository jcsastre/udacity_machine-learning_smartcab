<h1>Self-Driving Agent Report</h1>

<h2>Implementation of a Basic Driving Agent</h2>

As first task, we will move the smartcab around the environment using a 
random approach. The set of possible actions will be: None, forward, 
left, right. The deadline will be set to false, and we will observe how
the smartcab performs.

We can observe that in his random movement around the grid, from to time
it reaches the destination. We can also observe that always follow the
traffic rules, because the enviroment implementation code doesn't allow
any agent to violate these rules. But in this case it receives a reward
of -1.




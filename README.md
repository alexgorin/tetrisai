# AI for tetris
## Algorithm
Agents are described in agents.py.
The most effective - LimitedProbabilisticPlanningHierarchicalAgent - works the following way:
1. It is using utility function based of 4 features described in features.py
2. The weights for the features are calculated using a genetic algorithm (see training.py)
3. The agent considers combinations of movements (rotate till given orientation -> move to given position -> move all the way down) for 2 figures that are known at the moment.
4. For 10 pairs of moves with highest utilities (number 10 was rather arbitrary) and for every possible figure that might come next, calculates the utilities of possible outcomes and chooses the move with highest expected utility.

## The effectiveness
The top-performing algorithm was going past 50000 figures, so it would take long to evaluate its performance.
I might do it at some point.

## How to run
main.py - running the game GUI with the given agent
simulations.py - testing the agent without GUI
The code was written in Python 3.8. Might work in 3.7.  

# An Ensemble Method for Traffic Light Management

The main tool used to conduct experiments is [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) framework.

Ensemble methods used:
- Majority Voting
- Soft Voting
- Transformed Rank Voting
- Average Voting - Boltzmann Probs

**Reward** is defined as the change of the cumulative vehicle delay $r_{t} = D_{a_{t+1}} - D_{a_{t}}$
**Action**: Choose the next *Green* phase
**State Representation**: Vector of dimension $R^{\#GreenPhases+2*\#Lanes}$
**Environment**: Simulation of Urban MObility (SUMO)

## Our Approach

![Single Intersection](https://github.com/LucasAlegre/sumo-rl/blob/master/outputs/actions.png)

A single intersection consisted of:
- 2 Incoming - 2 Outgoing approaches
- Totally 8 Lanes
- 8 Permitted Movement Signals
- Sythetic Data built on SUMO:
    - Approach Length: $300m$
    - Cycle Traffic Plan Duration: $82s$
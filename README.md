# Pac-Man & Q-learning

A self-designed pacman agent that utilizes q-learning to compete in a capture the flag style game of Pac-Man.

The Game
--------

The adversarial game is a competition between team Read and team Blue, where each team consists of two Pac-Men all with the ability to turn into ghosts and back.  
The map is split in half where each half represents the home base of each team. While in their own half, an agent will be in the form of a ghost, and only when the agent crosses over to the opposing team's half do they turn into Pac-Man.  
The goal of each team is to eat all the pellets of the opposing team's side, while protecting their own base and pellets.

Our team consists of two agents; a defensive agent and an offensive agent.

Defensive Agent
---------------

The defensive agent is a simple reflex agent that takes score-maximizing actions. It's given features and weights that allow it to prioritize defensive actions over any other.  
Features included: # of teammates on team side, # of invades on team side, distance to closest invader etc. Each of these features is given a hard-coded weight to signifiy their importance. 

Due to these priorities, this agent tends to stay on the team's side of the map in ghost form.

Offensive Agent
---------------

The offensive agent uses q-learing to learn an optimal offensive policy over hundreds of games/training sessions. 

The policy changes this agent's focus to offensive features such as collecting pellets/capsules, avoiding ghosts, maximizing scores via eating pellets etc.  
These features are not supplied with weights at first, but after iterating through many games, q-learning allows this agent to figure out optimal weights for each feature, leading the agent to learn an optimal policy to win the game.

This agent tends to rush to the other team's half, staying as Pac-Man most of the time.

Build
-----
To watch the two agents in action, make sure you have Python 2.x and X11 installed (for graphics), and run
> python capture.py -r myTeam -b myTeam

This runs a game between two teams, each consisting of a defensive reflex and offensive q-learning agent.



# myTeam.py
# ---------

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'SmartOffense', second = 'HardDefense'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.
	"""
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class HardDefense(CaptureAgent):
	"""
	A simple reflex agent that takes score-maximizing actions. It's given 
	features and weights that allow it to prioritize defensive actions over any other.
	"""

	def registerInitialState(self, gameState):
		"""
		This method handles the initial setup of the
		agent to populate useful fields (such as what team
		we're on).
		"""

		CaptureAgent.registerInitialState(self, gameState)
		self.myAgents = CaptureAgent.getTeam(self, gameState)
		self.opAgents = CaptureAgent.getOpponents(self, gameState)
		self.myFoods = CaptureAgent.getFood(self, gameState).asList()
		self.opFoods = CaptureAgent.getFoodYouAreDefending(self, gameState).asList()

	# Finds the next successor which is a grid position (location tuple).
	def getSuccessor(self, gameState, action):
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			# Only half a grid position was covered
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	# Returns a counter of features for the state
	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	# Returns a dictionary of features for the state
	def getWeights(self, gameState, action):
		return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

	# Computes a linear combination of features and feature weights
	def evaluate(self, gameState, action):
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		return features * weights

	# Choose the best action for the current agent to take
	def chooseAction(self, gameState):
		agentPos = gameState.getAgentPosition(self.index)
		actions = gameState.getLegalActions(self.index)

		# Distances between agent and foods
		distToFood = []
		for food in self.myFoods:
			distToFood.append(self.distancer.getDistance(agentPos, food))

		# Distances between agent and opponents
		distToOps = []
		for opponent in self.opAgents:
			opPos = gameState.getAgentPosition(opponent)
			if opPos != None:
				distToOps.append(self.distancer.getDistance(agentPos, opPos))
		
		# Get the best action based on values
		values = [self.evaluate(gameState, a) for a in actions]
		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		return random.choice(bestActions)


class SmartOffense(CaptureAgent):
	"""
	The offensive agent uses q-learing to learn an optimal offensive policy 
	over hundreds of games/training sessions. The policy changes this agent's 
	focus to offensive features such as collecting pellets/capsules, avoiding ghosts, 
	maximizing scores via eating pellets etc.
	"""

	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)

		self.epsilon = 0.0 #exploration prob
		self.alpha = 0.2 #learning rate
		self.discountRate = 0.8
		self.weights = {'closest-food': -2.2558226236802597, 
										'bias': 1.0856704846852672, 
										'#-of-ghosts-1-step-away': -0.18419418670562, 
										'successorScore': -0.027287497346388308, 
										'eats-food': 9.970429654829946}
		"""
		Open weights file if it exists, otherwise start with empty weights.
		NEEDS TO BE CHANGED BEFORE SUBMISSION
		try:
			with open('weights.txt', "r") as file:
				self.weights = eval(file.read())
		except IOError:
				return
		"""


	#------------------------------- Q-learning Functions -------------------------------

	"""
	Iterate through all features (closest food, bias, ghost dist),
	multiply each of the features' value to the feature's weight,
	and return the sum of all these values to get the q-value.
	"""
	def getQValue(self, gameState, action):
		features = self.getFeatures(gameState, action)
		return features * self.weights

	"""
	Iterate through all q-values that we get from all
	possible actions, and return the highest q-value
	"""
	def getValue(self, gameState):
		qVals = []
		legalActions = gameState.getLegalActions(self.index)
		if len(legalActions) == 0:
			return 0.0
		else:
			for action in legalActions:
				qVals.append(self.getQValue(gameState, action))
			return max(qVals)

	"""
	Iterate through all q-values that we get from all
	possible actions, and return the action associated
	with the highest q-value.
	"""
	def getPolicy(self, gameState):
		values = []
		legalActions = gameState.getLegalActions(self.index)
		legalActions.remove(Directions.STOP)
		if len(legalActions) == 0:
			return None
		else:
			for action in legalActions:
				#self.updateWeights(gameState, action)
				values.append((self.getQValue(gameState, action), action))
		return max(values)[1]

	"""
	Calculate probability of 0.1.
	If probability is < 0.1, then choose a random action from
	a list of legal actions.
	Otherwise use the policy defined above to get an action.
	"""
	def chooseAction(self, gameState):
		# Pick Action
		legalActions = gameState.getLegalActions(self.index)
		action = None

		if len(legalActions) != 0:
			prob = util.flipCoin(self.epsilon)
			if prob:
				action = random.choice(legalActions)
			else:
				action = self.getPolicy(gameState)
		return action


	#------------------------------ Features And Weights --------------------------------

	# Define features to use. NEEDS WORK
	def getFeatures(self, gameState, action):
		# Extract the grid of food and wall locations
		food = gameState.getBlueFood()
		walls = gameState.getWalls()
		ghosts = []
		opAgents = CaptureAgent.getOpponents(self, gameState)
		# Get ghost locations and states if observable
		if opAgents:
			for opponent in opAgents:
				opPos = gameState.getAgentPosition(opponent)
				opIsPacman = gameState.getAgentState(opponent).isPacman
				if opPos and not opIsPacman: 
					ghosts.append(opPos)
		
		# Initialize features
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		# Successor Score
		features['successorScore'] = self.getScore(successor)

		# Bias
		features["bias"] = 1.0
		
		# compute the location of pacman after he takes the action
		x, y = gameState.getAgentPosition(self.index)
		dx, dy = Actions.directionToVector(action)
		next_x, next_y = int(x + dx), int(y + dy)
		
		# Number of Ghosts 1-step away
		features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
		# if there is no danger of ghosts then add the food feature
		if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
			features["eats-food"] = 1.0

		# Number of Ghosts scared
		#features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)
		
		# Closest food
		dist = self.closestFood((next_x, next_y), food, walls)
		if dist is not None:
			# make the distance a number less than one otherwise the update
			# will diverge wildly
			features["closest-food"] = float(dist) / (walls.width * walls.height) 

		# Normalize and return
		features.divideAll(10.0)
		return features

	"""
	Iterate through all features and for each feature, update
	its weight values using the following formula:
	w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
	"""
	def updateWeights(self, gameState, action):
		features = self.getFeatures(gameState, action)
		nextState = self.getSuccessor(gameState, action)

		# Calculate the reward. NEEDS WORK
		reward = nextState.getScore() - gameState.getScore()

		for feature in features:
			correction = (reward + self.discountRate*self.getValue(nextState)) - self.getQValue(gameState, action)
			self.weights[feature] = self.weights[feature] + self.alpha*correction * features[feature]


	#-------------------------------- Helper Functions ----------------------------------

	# Finds the next successor which is a grid position (location tuple).
	def getSuccessor(self, gameState, action):
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			# Only half a grid position was covered
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def closestFood(self, pos, food, walls):
		fringe = [(pos[0], pos[1], 0)]
		expanded = set()
		while fringe:
			pos_x, pos_y, dist = fringe.pop(0)
			if (pos_x, pos_y) in expanded:
				continue
			expanded.add((pos_x, pos_y))
			# if we find a food at this location then exit
			if food[pos_x][pos_y]:
				return dist
			# otherwise spread out from the location to its neighbours
			nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
			for nbr_x, nbr_y in nbrs:
				fringe.append((nbr_x, nbr_y, dist+1))
		# no food found
		return None

	# Update weights file at the end of each game
	#def final(self, gameState):
		#print self.weights
		#file = open('weights.txt', 'w')
		#file.write(str(self.weights))
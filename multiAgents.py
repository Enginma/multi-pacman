# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
import random
import util
from game import Agent
from math import inf


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        "Setting up function to find the distance between one position and pacman with manhattanDistance function."
        def distToPacman(position):
            return manhattanDistance(newPos, position)

        "Setting up function to find ghost position"
        def getGhostPos(ghost):
            dist = distToPacman(ghost.getPosition())

            "If the ghost is scared and not threatening pacman, return a positive value."
            if ghost.scaredTimer > dist:
                return inf


            "If the ghost is very close to Pacman, return a negative value to avoid it."
            if dist <= 1:
                return -inf

            return 0

        "Find the closest ghost and food position to pacman."
        closestGhost = min(map(getGhostPos, newGhostStates))
        closestFood = min(map(distToPacman, newFood.asList()), default = 0)

        "Recalculate the weight of each food with closest food having a larger weighting than farther foods."
        closestFood= 1.0 / (1.0 + closestFood)

        "Return a score using the current score + the ghost position + the reciprocal of important values, which is the distance to the food. "
        return successorGameState.getScore() + closestGhost + closestFood


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        "Format of result = [score, action, index, depth]. "
        result = self.getValue(gameState, 0, 0)
        action = result[1]
        return action

    "Returns the best move using max function. "
    def max(self, gameState, index, depth):

        legalMoves = gameState.getLegalActions(index)
        max_value = -inf
        max_action = ""

        for action in legalMoves:

            "Generate the successor state by applying the current action."
            successor = gameState.generateSuccessor(index, action)

            "Calculate the index and depth for the successor state. "
            successor_index = index + 1
            successor_depth = depth

            "After all agents take turn we go to next depth. "
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            "Get the successor's value only. "
            current_value, _ = self.getValue(successor, successor_index, successor_depth)

            "Update max value and action if current value is bigger than current max value. "
            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    "Returns the worst move using min function for the ghosts. "
    def min(self, gameState, index, depth):

        "Initialize min value and action and the legal moves. "
        legalMoves = gameState.getLegalActions(index)
        min_value = inf
        min_action = " "

        for action in legalMoves:

            "Generate the successor state by applying the current action."
            successor = gameState.generateSuccessor(index, action)

            "Calculate the index and depth for the successor state. "
            successor_index = index + 1
            successor_depth = depth

            "After all agents take turn we go to next depth. "
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            "Get the successor's value only. "
            current_value, _ = self.getValue(successor, successor_index, successor_depth)

            "Update min value and action if current value is smaller than current min value. "
            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action

    "Get value helper function for minimax. "
    def getValue(self, gameState, index, depth):

        "If game state is in terminal state then return no action"
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), " "

        "If index is 0 it is pacman's turn and we choose the pick the best move for it. "
        if index == 0:
            return self.max(gameState, index, depth)

        # Else it is the ghosts turn and we pick the pacman's worst move for it.
        else:
            return self.min(gameState, index, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        "Format of result = [score, action, index, depth, alpha, beta]"
        result = self.getValue(game_state, 0, 0, float("-inf"), float("inf"))
        action = result[1]
        return action

    "Return the max score move. "
    def max_value(self, game_state, index, depth, alpha, beta):

        "Initialize max value and actions and legal moves. "
        legalMoves = game_state.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""


        for action in legalMoves:

            "Generate the successor state by applying the current action."
            successor = game_state.generateSuccessor(index, action)

            "Calculate the index and depth for the successor state. "
            successor_index = index + 1
            successor_depth = depth

            "After all agents take turn we go to next depth. "
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            "Get the successor's action and value. "
            current_value, current_action = self.getValue(successor, successor_index, successor_depth, alpha, beta)

            "Update max_value and max_action for maximizer agent if current value is bigger than max value. "
            if current_value > max_value:
                max_value = current_value
                max_action = action

            "Update alpha value. "
            alpha = max(alpha, max_value)

            "Pruning values to make minimax faster. "
            if max_value > beta:
                return max_value, max_action

        return max_value, max_action

    "Returns the minimum score move. "
    def min_value(self, game_state, index, depth, alpha, beta):

        "Initialize max value and actions and legal moves. "
        legalMoves = game_state.getLegalActions(index)
        min_value = float("inf")
        min_action = ""


        for action in legalMoves:

            "Generate the successor state by applying the current action."
            successor = game_state.generateSuccessor(index, action)

            "Calculate the index and depth for the successor state. "
            successor_index = index + 1
            successor_depth = depth

            "After all agents take turn we go to next depth. "
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            "Get the successor's action and value. "
            current_value, current_action = self.getValue(successor, successor_index, successor_depth, alpha, beta)


            "Update min_value and min_action for minimizer agent if current value is smaller than min value. "
            if current_value < min_value:
                min_value = current_value
                min_action = action

            "Update beta value. "
            beta = min(beta, min_value)

            "Pruning values to make minimax faster. "
            if min_value < alpha:
                return min_value, min_action

        return min_value, min_action


    "Helper function for alpha-beta pruning"
    def getValue(self, game_state, index, depth, alpha, beta):

        "If game state is in terminal state then return no action"
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return game_state.getScore(), " "

        "For pacman we want best value and best action. "
        if index == 0:
            return self.max_value(game_state, index, depth, alpha, beta)

        # For ghosts we use worst value and worst action of pacman as it is its adversary.
        else:
            return self.min_value(game_state, index, depth, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        "If game state is in terminal state then return no action. "
        def expectimax(game_state, depth, index):
            if depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            "Get all legal actions. "
            legal_actions = game_state.getLegalActions(index)

            "If it is pacman's turn"
            if index == 0:
                max_value = float("-inf")

                "Get all legal actions and successors and set max value. "
                for action in legal_actions:
                    successor = game_state.generateSuccessor(index, action)
                    max_value = max(max_value, expectimax(successor, depth, index + 1))
                return max_value

           # "Ghosts turn"
            else:
                num_ghosts = game_state.getNumAgents() - 1
                expected_value = 0.0

                "Calculate the expected value among possible successor states. "
                for action in legal_actions:
                    successor = game_state.generateSuccessor(index, action)

                    "If all ghosts have taken their turns, decrease depth and switch to Pac-Man's turn"
                    if index == num_ghosts:
                        expected_value += expectimax(successor, depth - 1, 0)

                    else:
                        expected_value += expectimax(successor, depth, index + 1)

                return expected_value / len(legal_actions)

        "Get Pac-Man's legal actions from the initial game state. "
        legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float("-inf")

        "Evaluate the expected value of different actions for Pac-Man and choose the best one. "
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, self.depth, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    First we need to find the current pacman position, adversary's position, the current score, and how many
     food/capsule are left in the game. Then we make a list to store manhattan distance from pacman to each
     food positions. The we find the closest food to pacman and we find the reciporcal of the closest food
     to give it a weight to improve our evaluation function.
    """
    "*** YOUR CODE HERE ***"

    "Get the current position of Pacman and the positions of all the ghosts. "
    pacmanPos = currentGameState.getPacmanPosition()
    adversaryPos = currentGameState.getGhostPositions()

    "Get current game score. "
    gameScore = currentGameState.getScore()

    "Get the list of food positions, count the number of remaining food dots, and count the capsules. "
    foodList = currentGameState.getFood().asList()
    foodCount = len(foodList)
    capsuleCount = len(currentGameState.getCapsules())


    "List to store food positions from pacman. "
    foodDistances = []
    for food_position in foodList:
        distance = manhattanDistance(pacmanPos, food_position)
        foodDistances.append(distance)

    "Calculate the closest food distance (if there is still food left). "
    if foodDistances:
        closestFood = min(foodDistances)
    else:
        closestFood = 1

    "Iterate through each adversary position. "
    for i in range(len(adversaryPos)):
        danger = manhattanDistance(pacmanPos, adversaryPos[i])
        if danger < 3:
            closestFood = float('inf')
            break

    "Getting reciprocal of closest food to reevaluate importance of the given food. "
    inverseClosest = 1.0 / closestFood

    "Calculate the weighted sum of features. "
    "We give each "
    feature = [gameScore, inverseClosest, foodCount, capsuleCount]
    weights = [330, 10, -10, -10]
    result = 0
    for i in range(len(feature)):
        result += feature[i] * weights[i]

    return result


# Abbreviation
better = betterEvaluationFunction

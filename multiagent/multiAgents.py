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
from game import Directions
import random, util
import functools
from math import inf

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    _counter = 1

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        # food_dis, ghost_dis = maze_distance_to_closest_food_and_ghost(successorGameState)
        food_dis, ghost_dis = manhattan_distance_to_food_and_ghost(successorGameState)

        # print(successorGameState)
        # print(food_dis, ghost_dis)

        # if capsule is eaten chase ghosts
        if functools.reduce(lambda a, b: a and b, [a != 0 for a in [ghostState.scaredTimer for ghostState in
                                                                    currentGameState.getGhostStates()]]):
            manipulated_score = 1 / ghost_dis
        else:
            if ghost_dis < 2:
                return -9999
            # elif ghost_dis > 4 and food_dis < 2:
            #     manipulated_score = successorGameState.getScore()
            elif ghost_dis > 4:
                manipulated_score = 2 * successorGameState.getScore() - food_dis + ghost_dis / 2
            else:
                manipulated_score = 2 * successorGameState.getScore() - food_dis + ghost_dis

        return manipulated_score


def manhattan_distance_to_food_and_ghost(startState):
    food_positions = startState.getFood().asList()
    ghost_positions = [ghost.getPosition() for ghost in startState.getGhostStates()]
    pacman_position = startState.getPacmanPosition()

    try:
        closest_food = min([manhattanDistance(pacman_position, food) for food in food_positions])
    except ValueError:
        closest_food = 0
    try:
        closest_ghost = min([manhattanDistance(pacman_position, ghost) for ghost in ghost_positions])
    except ValueError:
        closest_ghost = 0

    return closest_food, closest_ghost


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.get_best_action(gameState)[0]

    def min_value(self, state, counter):
        agent_index = counter % state.getNumAgents()
        return min([self.state_value(s, counter + 1) for s in
                    [state.generateSuccessor(agent_index, action) for action in state.getLegalActions(agent_index)]])

    def max_value(self, state, counter):
        agent_index = counter % state.getNumAgents()
        return max([self.state_value(s, counter + 1) for s in
                    [state.generateSuccessor(agent_index, action) for action in state.getLegalActions(agent_index)]])

    def state_value(self, state, counter):

        is_terminal = state.isWin() or state.isLose() or self.depth * state.getNumAgents() <= counter

        if is_terminal:
            return self.evaluationFunction(state)
        elif counter % state.getNumAgents() == 0:
            return self.max_value(state, counter)
        else:
            return self.min_value(state, counter)

    def get_best_action(self, start_state):
        return max([(action, self.state_value(state, 1)) for action, state in
                    [(action, start_state.generateSuccessor(0, action)) for action in start_state.getLegalActions()]],
                   key=lambda item: item[1])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.get_best_action(gameState)[1]

    def min_value(self, state, alpha, beta, counter):
        agent_index = counter % state.getNumAgents()
        v = inf
        for action in state.getLegalActions(agent_index):
            successor = state.generateSuccessor(agent_index, action)
            v = min(v, self.state_value(successor, alpha, beta, counter + 1)[0])
            if v < alpha:
                return v,None
            beta = min(beta, v)
        return v,None

    def max_value(self, state, alpha, beta, counter):
        agent_index = counter % state.getNumAgents()
        max_action = None
        v = -inf
        for action in state.getLegalActions(agent_index):
            successor = state.generateSuccessor(agent_index, action)
            temp = v
            v = max(v, self.state_value(successor, alpha, beta, counter+1)[0])
            if v != temp:
                max_action = action
            if v > beta:
                return v,None
            alpha = max(alpha, v)
        return v,max_action

    def state_value(self, state, alpha, beta, counter):

        is_terminal = state.isWin() or state.isLose() or self.depth * state.getNumAgents() <= counter

        if is_terminal:
            return self.evaluationFunction(state),None
        elif counter % state.getNumAgents() == 0:
            return self.max_value(state, alpha, beta, counter)
        else:
            return self.min_value(state, alpha, beta, counter)

    def get_best_action(self, start_state):
        return self.state_value(start_state,-inf,inf,0)


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
        "*** YOUR CODE HERE ***"
        return self.get_best_action(gameState)[0]

    def exp_value(self, state, counter):
        agent_index = counter % state.getNumAgents()
        successor_states = [state.generateSuccessor(agent_index, action) for action in state.getLegalActions(agent_index)]
        return sum([1/len(successor_states)*self.state_value(s, counter + 1) for s in successor_states])

    def max_value(self, state, counter):
        agent_index = counter % state.getNumAgents()
        return max([self.state_value(s, counter + 1) for s in
                    [state.generateSuccessor(agent_index, action) for action in state.getLegalActions(agent_index)]])


    def state_value(self, state, counter):
        number_of_states = state.getNumAgents()
        is_terminal = state.isWin() or state.isLose() or self.depth * state.getNumAgents() <= counter

        if is_terminal:
            return self.evaluationFunction(state)
        elif counter % number_of_states == 0:
            return self.max_value(state, counter)
        else:
            return self.exp_value(state, counter)

    def get_best_action(self, start_state):
        return max([(action, self.state_value(state, 1)) for action, state in
                    [(action, start_state.generateSuccessor(0, action)) for action in start_state.getLegalActions()]],
                   key=lambda item: item[1])


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foods = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    capsule_amount = len(capsules)
    pos = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    ghost_poses = currentGameState.getGhostPositions()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    try:
        closest_food_distance = min([manhattanDistance(pos, food_pos) for food_pos in foods])
    except ValueError:
        closest_food_distance = 0

    try:
        closest_capsule_distance = min([manhattanDistance(pos, capsule_pos) for capsule_pos in capsules])
    except ValueError:
        closest_capsule_distance = 0

    try:
        closest_ghost_distance = min([manhattanDistance(pos, ghost_pos) for ghost_pos in ghost_poses])
    except ValueError:
        closest_ghost_distance = 0


    final_score = currentGameState.getScore()

    if functools.reduce(lambda a, b: a and b, [a != 0 for a in scared_times]):
        final_score = 2*currentGameState.getScore() -10*closest_ghost_distance
    elif closest_ghost_distance > 4:
        final_score = 2*currentGameState.getScore() - closest_food_distance
    elif closest_capsule_distance < closest_ghost_distance < 4:
        final_score = 2*currentGameState.getScore() -5*closest_capsule_distance
    elif closest_ghost_distance < 2:
        final_score = 2*currentGameState.getScore() + 10*closest_ghost_distance
    else:
        final_score = 2*currentGameState.getScore() - closest_food_distance - closest_capsule_distance + closest_ghost_distance


    return final_score


# Abbreviation
better = betterEvaluationFunction

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

from game import Agent

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        oldFood = currentGameState.getFood()
        ghostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # computa distância para o fantasma mais próximo.
        minDistanceGhost = float("+inf")

        for ghostPos in ghostPositions:
            minDistanceGhost = min(minDistanceGhost, util.manhattanDistance(newPos, ghostPos))

        # se a acao selecionada leva à colisão com o ghost, pontuação é mínima
        if minDistanceGhost == 0:
            return float("-inf")

        # se a acao conduzir para a vitoria, pontuação é máxima
        if successorGameState.isWin():
            return float("+inf")

        score = successorGameState.getScore()

        # incentiva acao que conduz o agente para mais longe do fantasma mais próximo
        score += 2 * minDistanceGhost

        minDistanceFood = float("+inf")

        for foodPos in newFoodList:
            minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

        # incentiva acao que conduz o agente para mais perto da comida mais próxima
        score -= 2 * minDistanceFood

        # incentiva acao que leva a uma comida
        if(successorGameState.getNumFood() < currentGameState.getNumFood()):
            score += 5

        # penaliza as acoes de parada
        if action == Directions.STOP:
            score -= 10

        return score


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

          Directions.STOP:
            The stop direction, which is always legal
    
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        minimax = self.minimax(gameState, agentIndex=0, depth=self.depth)
        return minimax['action']
  
    def minimax(self, gameState, agentIndex=0, depth='2', action=Directions.STOP):
        # Driver para as chamadas recursivas do algoritmo
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0: 
            depth = depth-1
        
        if gameState.isWin() or gameState.isLose() or depth == -1:
            return {'value':self.evaluationFunction(gameState), 'action':action}
        else:
            if agentIndex==0: 
                return self.maxValue(gameState,agentIndex,depth)
            else: 
                return self.minValue(gameState,agentIndex,depth)


    def maxValue(self, gameState, agentIndex, depth):
        # Escolhe o valor máximo entre dois estados de jogadas de um fantasma
        v = {'value':float('-inf'), 'action':Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)        

        for action in legalMoves:
            if action == Directions.STOP: 
                continue
            successorGameState = gameState.generateSuccessor(agentIndex, action) 
            successorMinMax = self.minimax(successorGameState, agentIndex+1, depth, action)

            if v['value'] <= successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action

        return v


    def minValue(self, gameState, agentIndex, depth):
        # Escolhe o valor mínimo entre dois estados de jogadas do Pacman
        v = {'value': float('inf'), 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:
            if action == Directions.STOP: 
                continue
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorMinMax = self.minimax(successorGameState, agentIndex+1, depth, action)

            if v['value'] >= successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
  Your minimax agent with alpha-beta pruning (question 3)
  """
  def getAction(self, gameState):
    """
    Returns the alpha-beta pruned minimax action using self.depth and self.evaluationFunction
    """
    _, action = self.minimax(gameState, self.index, self.depth, float("-inf"), float("+inf"))
    return action

  def maxValue(self, gameState, agentIndex, depth, alpha, beta):
    bestVal = float("-inf")
    legalMoves = gameState.getLegalActions(agentIndex)
    for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        valOfAction, _ = self.minimax(successorGameState, agentIndex+1, depth, alpha, beta)

        if max(bestVal, valOfAction) == valOfAction:
            bestVal, bestAction = valOfAction, action

        if bestVal > beta:
            return bestVal, bestAction
        alpha = max(alpha, bestVal)

    return bestVal, bestAction

  def minValue(self, gameState, agentIndex, depth, alpha, beta):
    bestVal = float("+inf")
    legalMoves = gameState.getLegalActions(agentIndex)
    for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        valOfAction, _ = self.minimax(successorGameState, agentIndex+1, depth, alpha, beta)

        if min(bestVal, valOfAction) == valOfAction:
            bestVal, bestAction = valOfAction, action

        if bestVal < alpha:
            return bestVal, bestAction

        beta = min(beta, bestVal)

    return bestVal, bestAction

  def minimax(self, gameState, agentIndex, depth, alpha, beta):
    agentIndex = agentIndex % gameState.getNumAgents()

    if agentIndex==0: depth = depth-1

    if gameState.isWin() or gameState.isLose() or depth == -1:
        return self.evaluationFunction(gameState), None

    if agentIndex == 0:
        bestVal, bestAction = self.maxValue(gameState, agentIndex, depth, alpha, beta)
    else:
        bestVal, bestAction = self.minValue(gameState, agentIndex, depth, alpha, beta)

    return bestVal, bestAction

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
        expectimax = self.expectimax(gameState, agentIndex=0, depth=self.depth)
        return expectimax['action']

    def expectimax(self, gameState, agentIndex=0, depth='2', action=Directions.STOP):
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0: 
            depth = depth-1
        
        if gameState.isWin() or gameState.isLose() or depth == -1:
            return {'value':self.evaluationFunction(gameState), 'action':action}
        else:
            if agentIndex==0: 
                return self.maxValue(gameState, agentIndex, depth)
            else: 
                return self.expValue(gameState, agentIndex, depth)
        
    def expValue(self, gameState, agentIndex, depth):
        v = {'value': 0, 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)        

        for action in legalMoves:
            if action == Directions.STOP: 
                continue
            successorGameState = gameState.generateSuccessor(agentIndex, action) 
            successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)
            probability = successorExpectiMax['value']/len(legalMoves)
            v['value']+= probability
        return v

    def maxValue(self, gameState, agentIndex, depth):
        v = {'value': float('-inf'), 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)        

        for action in legalMoves:
            if action == Directions.STOP: 
                continue
            successorGameState = gameState.generateSuccessor(agentIndex, action) 
            successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)

            if v['value'] <= successorExpectiMax['value']:
                v['value'] = successorExpectiMax['value']
                v['action'] = action
        return v

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
    """

    # prioriza o estado que leva à vitória
    if currentGameState.isWin():
        return float("+inf")

    # estado de derrota corresponde à pior avaliação
    if currentGameState.isLose():
        return float("-inf")

    # variáveis a serem usadas na cálculo da função de avaliação
    score = scoreEvaluationFunction(currentGameState)
    newFoodList = currentGameState.getFood().asList()
    newPos = currentGameState.getPacmanPosition()

    #
    # ATENÇÃO: variáveis não usadas AINDA! 
    # Procure modificar essa função para usar essas variáveis e melhorar a função de avaliação.
    # Descreva em seu relatório de que forma essas variáveis foram usadas.
    #

    # Recebe os estados dos fantasmas no jogo
    ghostStates = currentGameState.getGhostStates()

    # Inicializa uma lista para armazenar distâncias do Pacman para fantasmas assustados
    scaredGhostDistanceList = []

    # Inicializa uma lista para armazenar distâncias do Pacman para fantasmas não-assustados
    nonScaredGhostDistanceList = []

    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            # Calcula e armazena na lista a distância do Pacman para um fantasma assustado
            scaredGhostDistanceList.append(util.manhattanDistance(newPos, ghostState.getPosition()))
        else:
            # Calcula e armazena na lista a distância do Pacman para um fantasma não-assustado
            nonScaredGhostDistanceList.append(util.manhattanDistance(newPos, ghostState.getPosition()))
    
    minScaredGhostDistance = -1
    minNonScaredGhostDistance = -1

    if len(scaredGhostDistanceList) > 0:
        # Verifica se a lista é vazia
        minScaredGhostDistance = min(scaredGhostDistanceList)
    
    if len(nonScaredGhostDistanceList) > 0:
        # Verifica se a lista é vazia
        minNonScaredGhostDistance = min(nonScaredGhostDistanceList)
        
    # Incentiva o Pacman a perseguir fantasmas assustados
    score += 2 * minScaredGhostDistance

    # Incentiva o Pacman a fugir de fantasmas não-assustados
    score -= 4 * minNonScaredGhostDistance

    # calcula distância entre o agente e a pílula mais próxima
    minDistanceFood = float("+inf")

    for foodPos in newFoodList:
        minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

    # incentiva o agente a se aproximar mais da pílula mais próxima
    score -= 2 * minDistanceFood

    # incentiva o agente a comer pílulas 
    score -= 4 * len(newFoodList)

    # incentiva o agente a se mover para próximo das cápsulas
    capsulelocations = currentGameState.getCapsules()
    score -= 4 * len(capsulelocations)

    return score

# Abbreviation
better = betterEvaluationFunction

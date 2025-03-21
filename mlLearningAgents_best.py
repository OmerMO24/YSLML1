# # mlLearningAgents.py
# # parsons/27-mar-2017
# #
# # A stub for a reinforcement learning agent to work with the Pacman
# # piece of the Berkeley AI project:
# #
# # http://ai.berkeley.edu/reinforcement.html
# #
# # As required by the licensing agreement for the PacMan AI we have:
# #
# # Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# # 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).

# # This template was originally adapted to KCL by Simon Parsons, but then
# # revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# This version is good 
from __future__ import absolute_import
from __future__ import print_function
import random
from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from pacman_utils.util import manhattanDistance

class GameStateFeatures:
    def __init__(self, state: GameState):
        self.pacman_pos = state.getPacmanPosition()
        self.food = state.getFood()
        self.ghost_positions = state.getGhostPositions()
        
    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_pos == other.pacman_pos and 
                self.food == other.food and 
                self.ghost_positions == other.ghost_positions)
    
    def __hash__(self):
        return hash((self.pacman_pos, str(self.food), tuple(self.ghost_positions)))

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.9, 
                 maxAttempts: int = 30, numTraining: int = 10):
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_values = {}
        self.counts = {}
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        if endState.isWin():
            return 1000
        elif endState.isLose():
            return -200
        food_start = startState.getNumFood()
        food_end = endState.getNumFood()
        score_diff = endState.getScore() - startState.getScore()
        if food_end < food_start:
            # After eating last food, penalize ghost proximity
            if food_end == 0:
                ghost_dist = min(manhattanDistance(endState.getPacmanPosition(), g) 
                                 for g in endState.getGhostPositions())
                ghost_penalty = -50 / (ghost_dist + 1)  # Closer ghost = bigger penalty
                return score_diff + 50 + ghost_penalty
            return score_diff + 50
        return score_diff - 1

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures, legal_actions: list) -> float:
        q_vals = [self.getQValue(state, action) for action in legal_actions]
        return max(q_vals) if q_vals else 0.0

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, 
              nextState: GameStateFeatures, next_legal_actions: list):
        current_q = self.getQValue(state, action)
        max_next_q = self.maxQValue(nextState, next_legal_actions)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_values[(state, action)] = new_q

    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int, state: GameStateFeatures, action: Directions) -> float:
        bonus = 1.0 / (counts + 1)
        if state.food.count() == 1:
            last_food = [(x, y) for x in range(state.food.width) 
                         for y in range(state.food.height) if state.food[x][y]][0]
            dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}[action]
            next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            dist = manhattanDistance(next_pos, last_food)
            bonus += 50 / (dist + 1)  # Distance-based bonus
        return utility + bonus

    def getAction(self, state: GameState) -> Directions:
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        state_features = GameStateFeatures(state)
        
        if state_features.food.count() == 1:
            last_food = [(x, y) for x in range(state_features.food.width) 
                         for y in range(state_features.food.height) 
                         if state_features.food[x][y]][0]
            print(f"One food left at {last_food}, Pacman at {state_features.pacman_pos}")
            for action in legal:
                q_val = self.getQValue(state_features, action)
                adjusted_val = self.explorationFn(q_val, self.getCount(state_features, action), 
                                                state_features, action)
                print(f"Action {action}: Q={q_val}, Adjusted={adjusted_val}")
        
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            q_vals = [(self.explorationFn(self.getQValue(state_features, action), 
                                          self.getCount(state_features, action), 
                                          state_features, action), action) 
                      for action in legal]
            _, action = max(q_vals, key=lambda x: x[0]) if q_vals else (0, random.choice(legal))
        
        self.last_game_state = state
        self.last_state = state_features
        self.last_action = action
        self.updateCount(state_features, action)
        return action

    def final(self, state: GameState):
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        if self.last_state and self.last_action and self.last_game_state:
            reward = self.computeReward(self.last_game_state, state)
            next_state = GameStateFeatures(state)
            next_legal_actions = state.getLegalPacmanActions()
            if Directions.STOP in next_legal_actions:
                next_legal_actions.remove(Directions.STOP)
            self.learn(self.last_state, self.last_action, reward, next_state, next_legal_actions)
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print("Training Done (turning off epsilon and alpha)")
            print("-" * len("Training Done (turning off epsilon and alpha)"))
            self.setAlpha(0)
            self.setEpsilon(0)
        self.last_state = None
        self.last_action = None
        self.last_game_state = None


# This one is better: v2.0
# Going to test the gamma 
from __future__ import absolute_import
from __future__ import print_function
import random
from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from pacman_utils.util import manhattanDistance

class GameStateFeatures:
    def __init__(self, state: GameState):
        self.pacman_pos = state.getPacmanPosition()
        self.food = state.getFood()
        self.ghost_positions = state.getGhostPositions()
        
    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_pos == other.pacman_pos and 
                self.food == other.food and 
                self.ghost_positions == other.ghost_positions)
    
    def __hash__(self):
        return hash((self.pacman_pos, str(self.food), tuple(self.ghost_positions)))

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.9, 
                 maxAttempts: int = 30, numTraining: int = 10):
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_values = {}
        self.counts = {}
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        if endState.isWin():
            return 1000
        elif endState.isLose():
            return -200
        food_start = startState.getNumFood()
        food_end = endState.getNumFood()
        score_diff = endState.getScore() - startState.getScore()
        if food_end < food_start:
            if food_end == 0:
                ghost_dist = min(manhattanDistance(endState.getPacmanPosition(), g) 
                                 for g in endState.getGhostPositions())
                ghost_penalty = -200 / (ghost_dist + 1)  # Doubled penalty
                return score_diff + 200 + ghost_penalty
            return score_diff + 50
        return score_diff - 1

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures, legal_actions: list) -> float:
        q_vals = [self.getQValue(state, action) for action in legal_actions]
        return max(q_vals) if q_vals else 0.0

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, 
              nextState: GameStateFeatures, next_legal_actions: list):
        current_q = self.getQValue(state, action)
        max_next_q = self.maxQValue(nextState, next_legal_actions)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_values[(state, action)] = new_q

    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int) -> float:
        bonus = 1.0 / (counts + 1)
        if self.last_state and self.last_state.food.count() <= 1:  # Include post-food state
            bonus += 50
        return utility + bonus

    def getAction(self, state: GameState) -> Directions:
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        state_features = GameStateFeatures(state)
        
        # Debug when one food remains or just eaten
        if state_features.food.count() <= 1:
            last_food = [(x, y) for x in range(state_features.food.width) 
                         for y in range(state_features.food.height) 
                         if state_features.food[x][y]] or [(3, 3)]  # Default to last known if eaten
            print(f"Food count {state_features.food.count()} at {last_food[0]}, Pacman at {state_features.pacman_pos}")
            for action in legal:
                q_val = self.getQValue(state_features, action)
                adjusted_val = self.explorationFn(q_val, self.getCount(state_features, action))
                print(f"Action {action}: Q={q_val}, Adjusted={adjusted_val}")
        
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            q_vals = [(self.explorationFn(self.getQValue(state_features, action), 
                                          self.getCount(state_features, action)), action) 
                      for action in legal]
            _, action = max(q_vals, key=lambda x: x[0]) if q_vals else (0, random.choice(legal))
        
        self.last_game_state = state
        self.last_state = state_features
        self.last_action = action
        self.updateCount(state_features, action)
        return action

    def final(self, state: GameState):
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        if self.last_state and self.last_action and self.last_game_state:
            reward = self.computeReward(self.last_game_state, state)
            next_state = GameStateFeatures(state)
            next_legal_actions = state.getLegalPacmanActions()
            if Directions.STOP in next_legal_actions:
                next_legal_actions.remove(Directions.STOP)
            self.learn(self.last_state, self.last_action, reward, next_state, next_legal_actions)
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print("Training Done (turning off epsilon and alpha)")
            print("-" * len("Training Done (turning off epsilon and alpha)"))
            self.setAlpha(0)
            self.setEpsilon(0)
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

# This one got 8/10 80% of the time with the adjusted gamma
from __future__ import absolute_import
from __future__ import print_function
import random
from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from pacman_utils.util import manhattanDistance

class GameStateFeatures:
    def __init__(self, state: GameState):
        self.pacman_pos = state.getPacmanPosition()
        self.food = state.getFood()
        self.ghost_positions = state.getGhostPositions()
        
    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_pos == other.pacman_pos and 
                self.food == other.food and 
                self.ghost_positions == other.ghost_positions)
    
    def __hash__(self):
        return hash((self.pacman_pos, str(self.food), tuple(self.ghost_positions)))

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.6, 
                 maxAttempts: int = 30, numTraining: int = 10):
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_values = {}
        self.counts = {}
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        if endState.isWin():
            return 1000
        elif endState.isLose():
            return -200
        food_start = startState.getNumFood()
        food_end = endState.getNumFood()
        score_diff = endState.getScore() - startState.getScore()
        if food_end < food_start:
            if food_end == 0:
                ghost_dist = min(manhattanDistance(endState.getPacmanPosition(), g) 
                                 for g in endState.getGhostPositions())
                ghost_penalty = -200 / (ghost_dist + 1)
                return score_diff + 200 + ghost_penalty
            return score_diff + 50
        if food_end == 1:  # Discourage center lingering
            ghost_dist = min(manhattanDistance(endState.getPacmanPosition(), g) 
                             for g in endState.getGhostPositions())
            return score_diff - 25 / (ghost_dist + 1)
        return score_diff - 1

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures, legal_actions: list) -> float:
        q_vals = [self.getQValue(state, action) for action in legal_actions]
        return max(q_vals) if q_vals else 0.0

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, 
              nextState: GameStateFeatures, next_legal_actions: list):
        current_q = self.getQValue(state, action)
        max_next_q = self.maxQValue(nextState, next_legal_actions)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_values[(state, action)] = new_q

    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int, state: GameStateFeatures, action: Directions) -> float:
        bonus = 1.0 / (counts + 1)
        if state.food.count() == 2:  # Prioritize (1, 1) initially
            dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}[action]
            next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            dist_to_11 = manhattanDistance(next_pos, (1, 1))
            bonus += 50 / (dist_to_11 + 1)  # Guide to (1, 1)
        elif state.food.count() == 1:  # Then to remaining food
            last_food = [(x, y) for x in range(state.food.width) 
                         for y in range(state.food.height) if state.food[x][y]][0]
            dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}[action]
            next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            dist_to_food = manhattanDistance(next_pos, last_food)
            bonus += 50 / (dist_to_food + 1)  # Guide to (3, 3) or (1, 1)
        return utility + bonus

    def getAction(self, state: GameState) -> Directions:
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        state_features = GameStateFeatures(state)
        
        # Debug when food count is 2 or 1
        if state_features.food.count() <= 2:
            food_positions = [(x, y) for x in range(state_features.food.width) 
                              for y in range(state_features.food.height) 
                              if state_features.food[x][y]] or [(3, 3)]
            print(f"Food count {state_features.food.count()} at {food_positions}, Pacman at {state_features.pacman_pos}")
            for action in legal:
                q_val = self.getQValue(state_features, action)
                adjusted_val = self.explorationFn(q_val, self.getCount(state_features, action), 
                                                state_features, action)
                print(f"Action {action}: Q={q_val}, Adjusted={adjusted_val}")
        
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            q_vals = [(self.explorationFn(self.getQValue(state_features, action), 
                                          self.getCount(state_features, action), 
                                          state_features, action), action) 
                      for action in legal]
            _, action = max(q_vals, key=lambda x: x[0]) if q_vals else (0, random.choice(legal))
        
        self.last_game_state = state
        self.last_state = state_features
        self.last_action = action
        self.updateCount(state_features, action)
        return action

    def final(self, state: GameState):
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        if self.last_state and self.last_action and self.last_game_state:
            reward = self.computeReward(self.last_game_state, state)
            next_state = GameStateFeatures(state)
            next_legal_actions = state.getLegalPacmanActions()
            if Directions.STOP in next_legal_actions:
                next_legal_actions.remove(Directions.STOP)
            self.learn(self.last_state, self.last_action, reward, next_state, next_legal_actions)
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print("Training Done (turning off epsilon and alpha)")
            print("-" * len("Training Done (turning off epsilon and alpha)"))
            self.setAlpha(0)
            self.setEpsilon(0)
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

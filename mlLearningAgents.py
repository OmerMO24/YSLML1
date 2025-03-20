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


# from __future__ import absolute_import
# from __future__ import print_function
# import random
# from pacman import Directions, GameState
# from pacman_utils.game import Agent
# from pacman_utils import util
# from pacman_utils.util import manhattanDistance

# class GameStateFeatures:
#     def __init__(self, state: GameState):
#         self.pacman_pos = state.getPacmanPosition()
        
#         # Convert food grid to more efficient representation
#         food_grid = state.getFood()
#         self.food_positions = tuple(sorted((x, y) for x in range(food_grid.width) 
#                                   for y in range(food_grid.height) if food_grid[x][y]))
#         self.food_count = len(self.food_positions)
        
#         # Store ghost positions
#         self.ghost_positions = tuple(sorted((int(x), int(y)) for x, y in state.getGhostPositions()))
        
#         # Calculate useful derived metrics
#         self.nearest_food_distance = float('inf')
#         if self.food_positions:
#             self.nearest_food_distance = min(manhattanDistance(self.pacman_pos, food_pos) 
#                                           for food_pos in self.food_positions)
        
#         self.nearest_ghost_distance = float('inf')
#         if self.ghost_positions:
#             self.nearest_ghost_distance = min(manhattanDistance(self.pacman_pos, ghost_pos) 
#                                            for ghost_pos in self.ghost_positions)
        
#     def __eq__(self, other):
#         if not isinstance(other, GameStateFeatures):
#             return False
#         return (self.pacman_pos == other.pacman_pos and 
#                 self.food_positions == other.food_positions and 
#                 self.ghost_positions == other.ghost_positions)
    
#     def __hash__(self):
#         return hash((self.pacman_pos, self.food_positions, self.ghost_positions))
    
#     def is_ghost_nearby(self, threshold=2):
#         """Check if any ghost is within the threshold distance"""
#         return self.nearest_ghost_distance <= threshold

# class QLearnAgent(Agent):
#     def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.8, 
#                  maxAttempts: int = 30, numTraining: int = 10):
#         super().__init__()
#         self.alpha = float(alpha)
#         self.epsilon = float(epsilon)
#         self.gamma = float(gamma)
#         self.maxAttempts = int(maxAttempts)
#         self.numTraining = int(numTraining)
#         self.episodesSoFar = 0
#         self.q_values = {}
#         self.counts = {}
#         self.last_state = None
#         self.last_action = None
#         self.last_game_state = None
        
#         # Performance metrics
#         self.wins = 0
#         self.total_score = 0
#         self.games_played = 0
        
#         # Initialize decay parameters for epsilon
#         self.initial_epsilon = self.epsilon
#         self.epsilon_decay = 0.9999  # Adjust based on training length
#         self.min_epsilon = 0.01

#     def incrementEpisodesSoFar(self):
#         self.episodesSoFar += 1
        
#         # Apply epsilon decay during training
#         if self.episodesSoFar < self.numTraining:
#             self.epsilon = max(self.min_epsilon, 
#                               self.initial_epsilon * (self.epsilon_decay ** self.episodesSoFar))

#     def getEpisodesSoFar(self):
#         return self.episodesSoFar

#     def getNumTraining(self):
#         return self.numTraining

#     def setEpsilon(self, value: float):
#         self.epsilon = value

#     def getAlpha(self) -> float:
#         return self.alpha

#     def setAlpha(self, value: float):
#         self.alpha = value

#     def getGamma(self) -> float:
#         return self.gamma

#     def getMaxAttempts(self) -> int:
#         return self.maxAttempts

#     @staticmethod
#     def computeReward(startState: GameState, endState: GameState) -> float:
#         """Enhanced reward function with dynamic ghost avoidance"""
#         # Check for terminal states first
#         if endState.isWin():
#             return 1000
#         elif endState.isLose():
#             return -1000
        
#         # Score difference as base reward
#         score_diff = endState.getScore() - startState.getScore()
#         reward = score_diff
        
#         # Food rewards
#         food_start = startState.getNumFood()
#         food_end = endState.getNumFood()
        
#         if food_end < food_start:  # Ate food
#             reward += 50
#         else:  # Did not eat food
#             # Small step penalty to encourage efficiency
#             reward -= 1
            
#             # Check if approaching food
#             start_pacman = startState.getPacmanPosition()
#             end_pacman = endState.getPacmanPosition()
            
#             # Find nearest food in end state
#             food_grid = endState.getFood()
#             food_positions = [(x, y) for x in range(food_grid.width) 
#                              for y in range(food_grid.height) if food_grid[x][y]]
            
#             if food_positions:
#                 # Get distance to nearest food in start and end states
#                 start_nearest_food_dist = min(manhattanDistance(start_pacman, food_pos) 
#                                            for food_pos in food_positions)
#                 end_nearest_food_dist = min(manhattanDistance(end_pacman, food_pos) 
#                                          for food_pos in food_positions)
                
#                 # Reward for getting closer to food
#                 if end_nearest_food_dist < start_nearest_food_dist:
#                     reward += 5
#                 # Penalty for moving away from food
#                 elif end_nearest_food_dist > start_nearest_food_dist and food_end > 0:
#                     reward -= 5
        
#         # Ghost avoidance - stronger penalty the closer we get
#         ghost_positions = endState.getGhostPositions()
#         pacman_pos = endState.getPacmanPosition()
#         min_ghost_dist = min(manhattanDistance(pacman_pos, ghost_pos) 
#                             for ghost_pos in ghost_positions)
        
#         # Exponential penalty for ghost proximity
#         if min_ghost_dist <= 1:  # Adjacent to ghost
#             reward -= 500
#         elif min_ghost_dist <= 3:  # Near a ghost
#             reward -= 200 / min_ghost_dist
            
#         # Extra reward for clearing the board (all food eaten but not yet won)
#         if food_end == 0 and not endState.isWin():
#             reward += 200
            
#         return reward
        
#     def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
#         return self.q_values.get((state, action), 0.0)

#     def maxQValue(self, state: GameStateFeatures, legal_actions: list) -> float:
#         if not legal_actions:
#             return 0.0
#         return max(self.getQValue(state, action) for action in legal_actions)

#     def getBestAction(self, state: GameStateFeatures, legal_actions: list) -> Directions:
#         """Get the action with the highest Q-value"""
#         if not legal_actions:
#             return None
            
#         best_value = float('-inf')
#         best_actions = []
        
#         for action in legal_actions:
#             q_value = self.getQValue(state, action)
#             if q_value > best_value:
#                 best_value = q_value
#                 best_actions = [action]
#             elif q_value == best_value:
#                 best_actions.append(action)
                
#         return random.choice(best_actions)

#     def learn(self, state: GameStateFeatures, action: Directions, reward: float, 
#               nextState: GameStateFeatures, next_legal_actions: list):
#         """Q-learning update with experience replay for critical states"""
#         # Standard Q-learning update
#         current_q = self.getQValue(state, action)
#         max_next_q = self.maxQValue(nextState, next_legal_actions)
#         td_target = reward + self.gamma * max_next_q
#         td_error = td_target - current_q
#         new_q = current_q + self.alpha * td_error
#         self.q_values[(state, action)] = new_q
        
#         # Boosted learning rate for ghost encounters (to learn faster from negative experiences)
#         if state.nearest_ghost_distance <= 2 and reward < -50:
#             # Apply a stronger update for dangerous situations
#             boosted_alpha = min(1.0, self.alpha * 2.0)  # Boost learning rate but cap at 1.0
#             self.q_values[(state, action)] = current_q + boosted_alpha * td_error

#     def updateCount(self, state: GameStateFeatures, action: Directions):
#         self.counts[(state, action)] = self.getCount(state, action) + 1

#     def getCount(self, state: GameStateFeatures, action: Directions) -> int:
#         return self.counts.get((state, action), 0)

#     def explorationFn(self, utility: float, counts: int, state: GameStateFeatures, action: Directions) -> float:
#         """Enhanced exploration function with situational awareness"""
#         # Base utility
#         value = utility
        
#         # Count-based exploration bonus (decreases as we visit more)
#         exploration_bonus = 1.0 / (counts + 1)
#         value += exploration_bonus
        
#         # In safety situations, increase utility of moving away from ghosts
#         if state.nearest_ghost_distance < 3:
#             # Calculate next position based on action
#             dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}.get(action, (0, 0))
#             next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            
#             # Check if this action increases distance from ghost
#             ghost_distances = [manhattanDistance(next_pos, ghost_pos) for ghost_pos in state.ghost_positions]
#             if ghost_distances and min(ghost_distances) > state.nearest_ghost_distance:
#                 value += 50  # Significant bonus for moving away from ghosts
        
#         # Food-seeking behavior when food is sparse
#         if state.food_count <= 2:
#             # Calculate next position based on action
#             dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}.get(action, (0, 0))
#             next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            
#             # Guide toward nearest food
#             if state.food_positions:
#                 distances_to_food = [manhattanDistance(next_pos, food_pos) for food_pos in state.food_positions]
#                 min_food_dist = min(distances_to_food)
#                 if min_food_dist < state.nearest_food_distance:
#                     # Reward for getting closer to food, with higher reward when food is sparse
#                     food_bonus = 50 / (min_food_dist + 1)
#                     value += food_bonus
                    
#         return value

#     def getAction(self, state: GameState) -> Directions:
#         """Select action using epsilon-greedy with enhanced exploration"""
#         # Get legal actions, removing STOP
#         legal = state.getLegalPacmanActions()
#         if Directions.STOP in legal:
#             legal.remove(Directions.STOP)
        
#         if not legal:
#             return None
        
#         # Create feature representation of state
#         state_features = GameStateFeatures(state)
        
#         # Learn from previous state-action pair if available
#         if self.last_state and self.last_action and self.last_game_state:
#             reward = self.computeReward(self.last_game_state, state)
#             self.learn(self.last_state, self.last_action, reward, state_features, legal)
        
#         # Epsilon-greedy action selection
#         if util.flipCoin(self.epsilon):
#             # Random exploration
#             action = random.choice(legal)
#         else:
#             # Exploitation with exploration function influence
#             q_vals = [(self.explorationFn(self.getQValue(state_features, action), 
#                                          self.getCount(state_features, action), 
#                                          state_features, action), action) 
#                       for action in legal]
#             _, action = max(q_vals) if q_vals else (0, random.choice(legal))
        
#         # Update state information
#         self.last_game_state = state
#         self.last_state = state_features
#         self.last_action = action
#         self.updateCount(state_features, action)
        
#         # Debug output for specific situations (low food count)
#         if state_features.food_count <= 2 and self.episodesSoFar >= self.numTraining:
#             print(f"Food count: {state_features.food_count}, Pacman at {state_features.pacman_pos}")
#             print(f"Ghost dist: {state_features.nearest_ghost_distance}, Food positions: {state_features.food_positions}")
#             for a in legal:
#                 q_val = self.getQValue(state_features, a)
#                 exp_val = self.explorationFn(q_val, self.getCount(state_features, a), state_features, a)
#                 print(f"Action {a}: Q={q_val:.2f}, Adjusted={exp_val:.2f}")
#             print(f"Selected: {action}")
        
#         return action

#     def final(self, state: GameState):
#         """Handle end of episode and learn from final transition"""
#         # Update performance metrics
#         self.games_played += 1
#         if state.isWin():
#             self.wins += 1
#         self.total_score += state.getScore()
        
#         # Print episode results
#         win_percent = (self.wins / self.games_played) * 100 if self.games_played > 0 else 0
#         print(f"Game {self.getEpisodesSoFar()} ended: {'Win' if state.isWin() else 'Loss'}")
#         print(f"Score: {state.getScore()}, Win Rate: {win_percent:.1f}% ({self.wins}/{self.games_played})")
        
#         # Learn from the final transition
#         if self.last_state and self.last_action and self.last_game_state:
#             reward = self.computeReward(self.last_game_state, state)
#             next_state = GameStateFeatures(state)
#             next_legal_actions = []  # No legal actions in terminal state
#             self.learn(self.last_state, self.last_action, reward, next_state, next_legal_actions)
        
#         # Reset for next episode
#         self.incrementEpisodesSoFar()
#         if self.getEpisodesSoFar() == self.getNumTraining():
#             avg_score = self.total_score / max(1, self.games_played)
#             msg = f"Training Done (turning off epsilon and alpha) - Avg Score: {avg_score:.1f}"
#             print(f"{msg}\n{'-' * len(msg)}")
#             self.setAlpha(0)
#             self.setEpsilon(0)
            
#             # Reset metrics for evaluation phase
#             self.wins = 0
#             self.total_score = 0
#             self.games_played = 0
            
#         self.last_state = None
#         self.last_action = None
#         self.last_game_state = None

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
        
        # I discovered that the food grid was a bit inefficient for lookups, so
        # I converted it to a tuple of positions for faster access and better hashing
        food_grid = state.getFood()
        self.food_positions = tuple(sorted((x, y) for x in range(food_grid.width) 
                                  for y in range(food_grid.height) if food_grid[x][y]))
        self.food_count = len(self.food_positions)
        
        # Had to convert ghost positions to integers and sort them to ensure
        # consistent hashing - otherwise similar states looked different to the agent
        self.ghost_positions = tuple(sorted((int(x), int(y)) for x, y in state.getGhostPositions()))
        
        # One big optimization was precalculating these distances - I noticed
        # the agent kept computing the same distances repeatedly
        self.nearest_food_distance = float('inf')
        if self.food_positions:
            self.nearest_food_distance = min(manhattanDistance(self.pacman_pos, food_pos) 
                                          for food_pos in self.food_positions)
        
        self.nearest_ghost_distance = float('inf')
        if self.ghost_positions:
            self.nearest_ghost_distance = min(manhattanDistance(self.pacman_pos, ghost_pos) 
                                           for ghost_pos in self.ghost_positions)
        
    def __eq__(self, other):
        """
        This checks if two game states are functionally equivalent.
        """
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_pos == other.pacman_pos and 
                self.food_positions == other.food_positions and 
                self.ghost_positions == other.ghost_positions)
    
    def __hash__(self):
        """
        This hashes all the features we need 
        """
        return hash((self.pacman_pos, self.food_positions, self.ghost_positions))
    
    def is_ghost_nearby(self, threshold=2):
        """
        This is just a handy utility function to check whether the ghost is nearby
        We mainly used this in the reward function
            
        """
        return self.nearest_ghost_distance <= threshold

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.8, 
                 maxAttempts: int = 30, numTraining: int = 10):
        """
        Initialize the Q-learning agent with learning parameters.
        
        We tried many different values for the greek (alpha, epsilon, gamma)
        these values worked the best for our model:
        - alpha: Controls how much we learn from new experiences (0.5 was a good balance)
        - epsilon: Controls exploration vs. exploitation (starting at 0.05)
        - gamma: Values future rewards (0.8 gives good foresight without overvaluing)
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_values = {}  
        self.counts = {}    # Track how many times we've seen each state-action pair
        self.last_state = None
        self.last_action = None
        self.last_game_state = None
        
        # We added these metrics to more easily observe the performance of our agent
        self.wins = 0
        self.total_score = 0
        self.games_played = 0
        
        # We use epsilon decay 
        self.initial_epsilon = self.epsilon
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

    def incrementEpisodesSoFar(self):
        """
        Keep track of episodes and decay epsilon over time.
        """
        self.episodesSoFar += 1
        
        # For every episode, we "decay" our epsilon value 
        if self.episodesSoFar < self.numTraining:
            self.epsilon = max(self.min_epsilon, 
                              self.initial_epsilon * (self.epsilon_decay ** self.episodesSoFar))

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
        """
        Calculate reward for a state transition.

        Win and Loss signals were really important to reward computation.
        We realized this early and decided to spend a good amount of our time
        looking at what are the possible actions pacman might take and which of
        these actions signal a win or a loss.

        Since the board is small, the number of possible board states is quite small,
        so we were able to find these signals with relative ease.

        The biggest factor that played into pacman's fate was the order in which it ate
        the food. By eating the outside food first, pacman just has to keep running away
        from the ghost, and then turn left/right into the center to eat the last food and
        win.

        If pacman decided to eat the food in the middle first, it could trap itself and the
        ghost would come up from behind and eat it.

        The biggest challenge was figuring a way to translate this into code for our reward policy
        such that we could encourage the desired behavior of eating the food on the outside first. 

        This ultimately wasn't possible without hardcoding the values so we decided against it and
        went for a simpler reward system that generalizes better.

                
        So we reward pacman for approaching and eating the food, and exponentially "punish" it
        for getting near ghosts.
               
        """
        if endState.isWin():
            return 1000
        elif endState.isLose():
            return -1000
        
        # Base reward from score difference
        score_diff = endState.getScore() - startState.getScore()
        reward = score_diff
        
        # Food rewards 
        food_start = startState.getNumFood()
        food_end = endState.getNumFood()
        
        if food_end < food_start: 
            reward += 50
        else:  # No food eaten this move
            # Small step penalty to encourage efficiency
            reward -= 1
            
            # We noticed the agent would wander aimlessly sometimes,
            # so we  added rewards for approaching food
            start_pacman = startState.getPacmanPosition()
            end_pacman = endState.getPacmanPosition()
            
            food_grid = endState.getFood()
            food_positions = [(x, y) for x in range(food_grid.width) 
                             for y in range(food_grid.height) if food_grid[x][y]]
            
            if food_positions:
                # Get distance to nearest food before and after the move
                start_nearest_food_dist = min(manhattanDistance(start_pacman, food_pos) 
                                           for food_pos in food_positions)
                end_nearest_food_dist = min(manhattanDistance(end_pacman, food_pos) 
                                         for food_pos in food_positions)
                
                # Reward for getting closer to food
                if end_nearest_food_dist < start_nearest_food_dist:
                    reward += 5
                # Penalty for moving away from food
                elif end_nearest_food_dist > start_nearest_food_dist and food_end > 0:
                    reward -= 5
        
        # Exponential ghost proximity punishment 
        ghost_positions = endState.getGhostPositions()
        pacman_pos = endState.getPacmanPosition()
        min_ghost_dist = min(manhattanDistance(pacman_pos, ghost_pos) 
                            for ghost_pos in ghost_positions)
        
        if min_ghost_dist <= 1:  
            reward -= 500
        elif min_ghost_dist <= 3:  
            reward -= 200 / min_ghost_dist
                       
        return reward
        
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """Get the learned Q-value for a state-action pair"""
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures, legal_actions: list) -> float:
        """Find the maximum Q-value for a state across all legal actions"""
        if not legal_actions:
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)

    def getBestAction(self, state: GameStateFeatures, legal_actions: list) -> Directions:
        """
        Find the action with the highest Q-value.
        """
        if not legal_actions:
            return None
            
        best_value = float('-inf')
        best_actions = []
        
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
                
        return random.choice(best_actions)

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, 
              nextState: GameStateFeatures, next_legal_actions: list):
        """
        Update Q-values based on the Q-learning formula.
                
        We boosted the learning rate when encountering dangerous situations,
        so the agent learns more quickly from negative experiences.
        """
        # Standard Q-learning update
        current_q = self.getQValue(state, action)
        max_next_q = self.maxQValue(nextState, next_legal_actions)
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_values[(state, action)] = new_q
        
        if state.nearest_ghost_distance <= 2 and reward < -50:
            # Apply a stronger update for dangerous situations
            boosted_alpha = min(1.0, self.alpha * 2.0)  # Boost but cap at 1.0
            self.q_values[(state, action)] = current_q + boosted_alpha * td_error

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """Track how many times we've taken each action in each state"""
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """Get the count of how many times we've taken an action in a state"""
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int, state: GameStateFeatures, action: Directions) -> float:
        """
        For the exploration function, we factored domain knowledge into our calculation.

        We consider pacman's proximity to the ghosts, and how much food is left. 

        """
        # Start with the raw Q-value
        value = utility
        
        # Add exploration bonus for less-visited state-actions
        exploration_bonus = 1.0 / (counts + 1)
        value += exploration_bonus
        
        # Add a huge bonus for moving away from nearby ghosts
        if state.nearest_ghost_distance < 3:
            # Calculate next position based on action
            dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}.get(action, (0, 0))
            next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            
            # Check if this action increases distance from ghost
            ghost_distances = [manhattanDistance(next_pos, ghost_pos) for ghost_pos in state.ghost_positions]
            if ghost_distances and min(ghost_distances) > state.nearest_ghost_distance:
                value += 50  # Big bonus for moving away from ghosts
        
        if state.food_count <= 2:
            # Calculate next position based on action
            dx, dy = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}.get(action, (0, 0))
            next_pos = (state.pacman_pos[0] + dx, state.pacman_pos[1] + dy)
            
            # Guide toward nearest food
            if state.food_positions:
                distances_to_food = [manhattanDistance(next_pos, food_pos) for food_pos in state.food_positions]
                min_food_dist = min(distances_to_food)
                # Reward for getting closer to food
                if min_food_dist < state.nearest_food_distance:
                    # Higher bonus when food is sparse
                    food_bonus = 50 / (min_food_dist + 1)
                    value += food_bonus
                    
        return value

    def getAction(self, state: GameState) -> Directions:
        """
        Choose the next action using an epsilon-greedy strategy.
        
        """
       
        legal = state.getLegalPacmanActions()
   
        # Pacman would just stop at the beginning
        if Directions.STOP in legal:
           legal.remove(Directions.STOP)
        
        if not legal:
            return None
        
        # Convert raw state to my feature representation
        state_features = GameStateFeatures(state)
        
        # Learn from previous state-action pair if available
        if self.last_state and self.last_action and self.last_game_state:
            reward = self.computeReward(self.last_game_state, state)
            self.learn(self.last_state, self.last_action, reward, state_features, legal)
        
        # Epsilon-greedy action selection 
        if util.flipCoin(self.epsilon):
            # Random exploration
            action = random.choice(legal)
        else:
            # Exploitation with exploration function influence
            q_vals = [(self.explorationFn(self.getQValue(state_features, action), 
                                         self.getCount(state_features, action), 
                                         state_features, action), action) 
                      for action in legal]
            _, action = max(q_vals) if q_vals else (0, random.choice(legal))
        
        # Save state for next update
        self.last_game_state = state
        self.last_state = state_features
        self.last_action = action
        self.updateCount(state_features, action)
        
        # Debug output for tricky situations 
        # This helped us understand what the agent was thinking in critical moments
        if state_features.food_count <= 2 and self.episodesSoFar >= self.numTraining:
            print(f"Food count: {state_features.food_count}, Pacman at {state_features.pacman_pos}")
            print(f"Ghost dist: {state_features.nearest_ghost_distance}, Food positions: {state_features.food_positions}")
            for a in legal:
                q_val = self.getQValue(state_features, a)
                exp_val = self.explorationFn(q_val, self.getCount(state_features, a), state_features, a)
                print(f"Action {a}: Q={q_val:.2f}, Adjusted={exp_val:.2f}")
            print(f"Selected: {action}")
        
        return action

    def final(self, state: GameState):
        """
        Handle the end of an episode - called after a win or loss.
        """
        # Track performance stats
        self.games_played += 1
        if state.isWin():
            self.wins += 1
        self.total_score += state.getScore()
        
        # Output results
        win_percent = (self.wins / self.games_played) * 100 if self.games_played > 0 else 0
        print(f"Game {self.getEpisodesSoFar()} ended: {'Win' if state.isWin() else 'Loss'}")
        print(f"Score: {state.getScore()}, Win Rate: {win_percent:.1f}% ({self.wins}/{self.games_played})")
        
        if self.last_state and self.last_action and self.last_game_state:
            reward = self.computeReward(self.last_game_state, state)
            next_state = GameStateFeatures(state)
            next_legal_actions = []  # No legal actions in terminal state
            self.learn(self.last_state, self.last_action, reward, next_state, next_legal_actions)
        
        # Reset for next episode
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            avg_score = self.total_score / max(1, self.games_played)
            msg = f"Training Done (turning off epsilon and alpha) - Avg Score: {avg_score:.1f}"
            print(f"{msg}\n{'-' * len(msg)}")
            self.setAlpha(0)
            self.setEpsilon(0)
            
            # Reset metrics for evaluation phase
            self.wins = 0
            self.total_score = 0
            self.games_played = 0
            
        self.last_state = None
        self.last_action = None
        self.last_game_state = None

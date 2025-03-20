
# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

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
        
        # The food grid was a bit inefficient for lookups, so
        # we converted it to a tuple of positions for faster access
        food_grid = state.getFood()
        food_list = []
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    food_list.append((x, y))
        food_list.sort()
        self.food_positions = tuple(food_list)
        self.food_count = len(self.food_positions)
        
        # Had to convert ghost positions to integers and sort them to ensure
        # consistent hashing - otherwise similar states looked different to the agent
        ghost_list = []
        for ghost_pos in state.getGhostPositions():
            x, y = ghost_pos
            ghost_list.append((int(x), int(y)))
        ghost_list.sort()
        self.ghost_positions = tuple(ghost_list)
        
        # The nearest food distance is needed in multiple functions, so we
        # lifted the value to be part of the game state and cache it so it
        # only needs to be calculated once 
        self.nearest_food_distance = float('inf')
        if self.food_positions:
            min_food_dist = float('inf')
            for food_pos in self.food_positions:
                dist = manhattanDistance(self.pacman_pos, food_pos)
                if dist < min_food_dist:
                    min_food_dist = dist
            self.nearest_food_distance = min_food_dist

        # The same philosophy behind caching the nearest food distance,
        # except this time we cache the distance of the nearest ghost
        self.nearest_ghost_distance = float('inf')
        if self.ghost_positions:
            min_ghost_dist = float('inf')
            for ghost_pos in self.ghost_positions:
                dist = manhattanDistance(self.pacman_pos, ghost_pos)
                if dist < min_ghost_dist:
                    min_ghost_dist = dist
            self.nearest_ghost_distance = min_ghost_dist
        
        # Store the original state for legal action access
        self.state = state
        
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
    
    def getLegalActions(self):
        """Get legal actions for this state"""
        if not hasattr(self, 'state') or self.state is None:
            return []
            
        actions = self.state.getLegalPacmanActions()
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.05, gamma: float = 0.8, 
                 maxAttempts: int = 30, numTraining: int = 10):
        """
        Initialize the Q-learning agent with learning parameters.
        
        We tried many different values for the greek (alpha, epsilon, gamma)
        these values worked the best for our model. Just as a reminder:
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


        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
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
            food_positions = []
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if food_grid[x][y]:
                        food_positions.append((x, y))
            
            if food_positions:
                # Get distance to nearest food before and after the move
                start_nearest_food_dist = float('inf')
                for food_pos in food_positions:
                    dist = manhattanDistance(start_pacman, food_pos)
                    if dist < start_nearest_food_dist:
                        start_nearest_food_dist = dist
                
                end_nearest_food_dist = float('inf')
                for food_pos in food_positions:
                    dist = manhattanDistance(end_pacman, food_pos)
                    if dist < end_nearest_food_dist:
                        end_nearest_food_dist = dist
                
                # Reward for getting closer to food
                if end_nearest_food_dist < start_nearest_food_dist:
                    reward += 5
                # Penalty for moving away from food
                elif end_nearest_food_dist > start_nearest_food_dist and food_end > 0:
                    reward -= 5
        
        # Exponential ghost proximity punishment 
        ghost_positions = endState.getGhostPositions()
        pacman_pos = endState.getPacmanPosition()
        min_ghost_dist = float('inf')
        for ghost_pos in ghost_positions:
            dist = manhattanDistance(pacman_pos, ghost_pos)
            if dist < min_ghost_dist:
                min_ghost_dist = dist
        
        if min_ghost_dist <= 1:  
            reward -= 500
        elif min_ghost_dist <= 3:  
            reward -= 200 / min_ghost_dist
                       
        return reward
        
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Get the learned Q-value for a state-action pair

        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)

        """
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Find the maximum Q-value for a state across all legal actions

        Args:
            state: The given state
 
        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legal_actions = state.getLegalActions() if hasattr(state, 'getLegalActions') else []
        
        if not legal_actions:
            return 0.0
            
        max_q_value = float('-inf')
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                
        return max_q_value if max_q_value != float('-inf') else 0.0

    def getBestAction(self, state: GameStateFeatures) -> Directions:
        """
        Find the action with the highest Q-value.
        """
        legal_actions = state.getLegalActions() if hasattr(state, 'getLegalActions') else []
        
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

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        """
        Update Q-values based on the Q-learning formula.
                
        We boosted the learning rate when encountering dangerous situations,
        so the agent learns more quickly from negative experiences.
        
        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Standard Q-learning update
        current_q = self.getQValue(state, action)
        max_next_q = self.maxQValue(nextState)
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_values[(state, action)] = new_q
        
        if state.nearest_ghost_distance <= 2 and reward < -50:
            # Apply a stronger update for dangerous situations
            boosted_alpha = min(1.0, self.alpha * 2.0)  # Boost but cap at 1.0
            self.q_values[(state, action)] = current_q + boosted_alpha * td_error

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Get the count of how many times we've taken an action in a state

        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state

        """
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts
                
        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # Add exploration bonus for less-visited state-actions
        exploration_bonus = 1.0 / (counts + 1)
        return utility + exploration_bonus

    def getAction(self, state: GameState) -> Directions:
        """
        Choose the next action using an epsilon-greedy strategy.

        We factor the current game state into the calculation:
        - Distance to ghost
        - Distance to food
        - Amount of food left
            
        Args:
            state: the current state

        Returns:
            The action to take
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
            self.learn(self.last_state, self.last_action, reward, state_features)
        
        # Epsilon-greedy action selection 
        if util.flipCoin(self.epsilon):
            # Random exploration
            action = random.choice(legal)
        else:
            # Exploitation with exploration function influence
            q_vals = []
            for action in legal:
                # Get base exploration value from explorationFn
                q_value = self.getQValue(state_features, action)
                count = self.getCount(state_features, action)
                base_value = self.explorationFn(q_value, count)
                
                # Apply domain-specific bonuses
                final_value = base_value
                
                # Add a huge bonus for moving away from nearby ghosts
                if state_features.nearest_ghost_distance < 3:
                    # Calculate next position based on action
                    dx = 0
                    dy = 0
                    if action == 'North':
                        dy = 1
                    elif action == 'South':
                        dy = -1
                    elif action == 'East':
                        dx = 1
                    elif action == 'West':
                        dx = -1
                        
                    next_pos = (state_features.pacman_pos[0] + dx, state_features.pacman_pos[1] + dy)
                    
                    # Check if this action increases distance from ghost
                    ghost_distances = []
                    for ghost_pos in state_features.ghost_positions:
                        dist = manhattanDistance(next_pos, ghost_pos)
                        ghost_distances.append(dist)
                        
                    if ghost_distances and min(ghost_distances) > state_features.nearest_ghost_distance:
                        final_value += 50  # Big bonus for moving away from ghosts
                
                if state_features.food_count <= 2:
                    # Calculate next position based on action
                    dx = 0
                    dy = 0
                    if action == 'North':
                        dy = 1
                    elif action == 'South':
                        dy = -1
                    elif action == 'East':
                        dx = 1
                    elif action == 'West':
                        dx = -1
                        
                    next_pos = (state_features.pacman_pos[0] + dx, state_features.pacman_pos[1] + dy)
                    
                    # Guide toward nearest food
                    if state_features.food_positions:
                        distances_to_food = []
                        for food_pos in state_features.food_positions:
                            dist = manhattanDistance(next_pos, food_pos)
                            distances_to_food.append(dist)
                            
                        min_food_dist = min(distances_to_food) if distances_to_food else float('inf')
                        # Reward for getting closer to food
                        if min_food_dist < state_features.nearest_food_distance:
                            # Higher bonus when food is sparse
                            food_bonus = 50 / (min_food_dist + 1)
                            final_value += food_bonus
                
                q_vals.append((final_value, action))
                
            if q_vals:
                max_val = q_vals[0][0]
                max_action = q_vals[0][1]
                for val, act in q_vals:
                    if val > max_val:
                        max_val = val
                        max_action = act
                action = max_action
            else:
                action = random.choice(legal)
        
        # Save state for next update
        self.last_game_state = state
        self.last_state = state_features
        self.last_action = action
        self.updateCount(state_features, action)
                       
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
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
            self.learn(self.last_state, self.last_action, reward, next_state)
        
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

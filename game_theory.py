import numpy as np
import random

# Define the game environment
class SimpleGame:
    def __init__(self):
        # Payoff matrix
        self.payoffs = {
            (0, 0): (2, 2),
            (0, 1): (0, 3),
            (1, 0): (3, 0),
            (1, 1): (1, 1)
        }

    def step(self, action1, action2):
        """Take actions for both players and return their respective rewards."""
        return self.payoffs[(action1, action2)]

class QLearningAgentEnhanced:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = np.zeros(n_actions)
        self.n_actions = n_actions
        self.lr = learning_rate
        self.df = discount_factor
        # Separate exploration rates for each action
        self.er = np.full(n_actions, exploration_rate)
        self.global_er = exploration_rate
        self.ed = exploration_decay

    def choose_action(self):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < np.max(self.er):
            return random.choice(range(self.n_actions))
        return np.argmax(self.q_table)

    def learn(self, action, reward):
        """Update Q-values using Q-learning update rule."""
        best_next_action = np.argmax(self.q_table)
        td_target = reward + self.df * self.q_table[best_next_action]
        td_error = td_target - self.q_table[action]
        self.q_table[action] += self.lr * td_error

        # Decay exploration rate for the taken action
        self.er[action] *= self.ed
        # Occasionally reset global exploration rate to ensure continued exploration
        if episode % 1000 == 0:
            self.global_er *= self.ed
            self.er = np.maximum(self.er, self.global_er)

# Training parameters
n_episodes = 999999

# Re-initialize game and agents
game = SimpleGame()
agent1_enhanced = QLearningAgentEnhanced(n_actions=2)
agent2_enhanced = QLearningAgentEnhanced(n_actions=2)

# Train the enhanced agents
for episode in range(n_episodes):
    # Agents decide on an action
    action1 = agent1_enhanced.choose_action()
    action2 = agent2_enhanced.choose_action()

    # Get rewards from the game
    reward1, reward2 = game.step(action1, action2)

    # Agents learn from the outcomes
    agent1_enhanced.learn(action1, reward1)
    agent2_enhanced.learn(action2, reward2)

agent1_enhanced.q_table, agent2_enhanced.q_table

print(agent1_enhanced.q_table, agent2_enhanced.q_table)

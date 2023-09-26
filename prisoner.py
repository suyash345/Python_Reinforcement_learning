
import numpy as np

# Actions: 0 = Cooperate, 1 = Betray
n_actions = 2

# States: 0 = Both Cooperate, 1 = Both Betray, 2 = I Cooperate & Opponent Betrays, 3 = I Betray & Opponent Cooperates
n_states = 4

# Q-table initialization
Q_table1 = np.zeros([n_states, n_actions])

Q_table2 = np.zeros([n_states, n_actions])

# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 50000

# Rewards matrix
Rewards_matrix = np.array([[1, 3],
              [2, 2],
              [3, 0],
              [0, 3]])


ep_rewards = []
aggr_ep_rewards = {"ep":[],"avg":[],"min":[],"max":[]}


for episode in range(episodes):
    state = np.random.choice(n_states)
    if np.random.random() > epsilon:
        action = np.argmax(Q_table1[state])
    else:
        action = np.random.choice(n_actions)
    opponent_state = np.random.choice(n_states  )
    if np.random.random() > epsilon:
        opponent_action = np.argmax(Q_table2[opponent_state])
    else:
        opponent_action = np.random.choice(n_actions)

    if action == 0 and opponent_action == 0:
        next_state = 0
    elif action == 1 and opponent_action == 1:
        next_state = 1
    elif action == 0 and opponent_action == 1:
        next_state = 2
    else:  # action == 1 and opponent_action == 0
        next_state = 3

# Get reward from reward matrix
    reward_player1 = Rewards_matrix[state, action]
    reward_player2 = Rewards_matrix[opponent_state, opponent_action]
# Q-learning update
    Q_table1[state, action] = Q_table1[state, action] + alpha * (reward_player1 + gamma * np.max(Q_table1[next_state, :]) - Q_table1[state, action])

    Q_table2[opponent_state, opponent_action] = Q_table2[opponent_state, opponent_action] + alpha * (reward_player2 + gamma * np.max(Q_table2[next_state, :]) - Q_table2[opponent_state, opponent_action])

# Display the trained Q-table
nash_action1 = np.argmax(Q_table1, axis=1)
nash_action2 = np.argmax(Q_table2, axis=1)

print(nash_action1, "/" , nash_action2)
print(Q_table1,"/",Q_table2)
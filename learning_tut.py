# the number of new states will be too much. Therefore, we need to turn the continous variables to discrete variables
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
state = env.reset()
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # how important are future actions over current reward.
EPISODES = 25000
SHOW_EVERY = 2000


epsilion = 0.5 # the chance of random actions
START_EPSILION_DECAY = 1
END_EPSILION_DECAY = EPISODES//2
epsilion_decay_value  = epsilion/(END_EPSILION_DECAY - START_EPSILION_DECAY)

# print(env.observation_space.high) # might not know these values
# print(env.observation_space.low)
# print(env.action_space.n) # the number of actions we can take.

DISCRETE_OS_SIZE = [20, 20] # hardcoded ""bucket"  from the high to low we want to seperate the values into 20 "chunks/buckets" in this case it is a (x,y) coordiante, its like creating a 20*20 grid

discrete_os_win_size = (env.observation_space.high  - env.observation_space.low)/DISCRETE_OS_SIZE    # the size of bucket. the [a,b] where a is the position of car, and b is the velocity of car.

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))   #q table of rewards related to the combination of moves. 20*20*3 table, 20*20 is the number of combinations with the state of the env,observations,3 is the number of actions.

ep_rewards = []
aggr_ep_rewards = {"ep":[],"avg":[],"min":[],"max":[]}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    episode_reward = 0
    if episode%SHOW_EVERY == 0:
        print(episode)
    discrete_state = get_discrete_state(state[0])
    done = False
    while not done:
        if np.random.random() > epsilion:
            action = np.argmax(q_table[discrete_state]) # argmax returns the position  of the max value
        else:
action = np.random.randint(0,env.action_space.n)

        new_state, reward, done, truncated, info = env.step(action) # reward here is -1, and 0 when it hits the flag
        episode_reward +=reward
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # this returns the actual q value not index
            current_q_value = q_table[discrete_state + (action,)]
            new_q = (1-LEARNING_RATE)*current_q_value + LEARNING_RATE * (reward+DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] = new_q # update q table
        elif new_state[0] >= env.goal_position:
            #print(f"We made it on epiosde {episode}")
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
    if END_EPSILION_DECAY >= episode >= START_EPSILION_DECAY:
        epsilion -= epsilion_decay_value
    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Epsiode: {episode} Average: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])} ")



env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()
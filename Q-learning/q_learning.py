# %%
import random
import re
from typing import List, Tuple

import numpy as np

# %%
# Q-learning is model based (no prior knowlegde of the env)
# based on Markov decision process
# used in situation with finite actions, states and steps
# works by exploring every action at every state and evaluates by assigning
# it a Q-Value
# Q-table stores best action at any state (main brain)


# %%
grid_size = 3
start = (0, 0)
goal = (2, 2)
obstacle = (1, 1)
#
actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),  # up  # down  # left  # right
]


# %%
# ensures we're within grid boundaries, and didnt run into an opstacle
def is_valid_state(state: Tuple[int, int]) -> bool:
    return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size and state != obstacle


# %%
# next state func to move from a given state toward a direction
# and checks if new state is valid
def get_next_state(state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
    new_state = (state[0] + action[0], state[1] + action[1])
    return new_state if is_valid_state(new_state) else state


# %%
# Q-Learning params
# explr_rate: exploration 30% vs exploitation param 70%
explr_rate = 0.3
#
# lr: for learning rate, how much of teh new info is kept
# lr = 0.3 -> incorporate 30% of new info and retain 70% of old info when
# updating q-value
lr = 0.3
#
# disct_fct is the discount factor, gamma = 0.99 -> future estimated rewards are
# valuated at %99 of their actual value
# immediate rewards are considered a little more
disct_fct = 0.99
episodes = 10000


# %%
# Rewards system: some feedback signal that inform the agent whether the actions
# are good or bad,
# so reward arriving at goal, and penalize hittig obstacles
# get reward formula: goal-> +100, obstacle or wall -> -10, for each step -> -1 (to find
# the best path)
def get_reward(state: Tuple[int, int], next_state: Tuple[int, int]) -> int:
    if next_state == goal:
        return 100
    elif (next_state == obstacle) or (next_state == state):
        return -10
    else:
        return -1


# %%
# choose action determines whetehr the next move will be an exploratory move,
# or a conservative one picked from the q-table, this is the epsilon-greedy
# strategy
# it assured continous learning even when the agent finds a good policy
# returns a move (up, down...)
def choose_action(state: Tuple[int, int], q_table: np.ndarray) -> Tuple[int, int]:
    if random.uniform(0, 1) < explr_rate:
        return random.choice(actions)
    else:
        return actions(np.argmax(q_table[state]))


# %%
# q value of each action at a given state is calculated using belman equation
# disct_fct * np.max(q_table[next_state]) is to factor in the future reward
# so best action is the one that maximizes both the immediate & future reward
def update_qtable(
    q_table: np.ndarray,
    state: Tuple[int, int],
    action: Tuple[int, int],
    next_state: Tuple[int, int],
    reward: int,
):
    action_index = actions.index(action)
    q_table[state][action_index] += lr * (
        reward + disct_fct * np.max(q_table[next_state]) - q_table[state][action_index]
    )


# %%

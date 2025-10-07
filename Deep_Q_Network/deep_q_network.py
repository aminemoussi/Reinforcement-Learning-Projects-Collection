# %%
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

# %%
# creating the environement
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("feartures of a space: ", state_size)  # 4 features
print("actions possible: ", action_size)  # 2 actions, (left or right)


# %%
# the model takes in 4features pf a state, has 2 hiddne layes each with 24 neurons
# and the last layer outputs 2 neurons that represent the q-value of each possible actions
# giving a total of 770 learnable params


# %%

from __future__ import division

import gym
import tflearn
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from statistics import mean, median
from tflearn.layers.estimator import regression
from collections import Counter

env = gym.make('SpaceInvaders-v0')
init_state = env.reset()
env.render()

input('press enter to exit')


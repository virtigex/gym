from __future__ import division

import gym
import tflearn
import numpy as np
import pickle
import time

from tflearn.layers.core import input_data, dropout, fully_connected
from statistics import mean, median
from tflearn.layers.estimator import regression
from collections import Counter

env = gym.make('SpaceInvaders-v0')
init_state = env.reset()

goal_steps = 10000
#initial_games = 10000
initial_games = 20
score_requirement = 300
INPUT_MAX = 5

def ran_game():
    for ep in range(5):
        print('ep = {}'.format(ep))
        env.reset()
        score = 0
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            score += reward
            if done or info['ale.lives'] < 3:
                break
        print(t, reward)

#acts = []
#for _ in range(1000):
#    acts.append(env.action_space.sample())
#print(acts)

def act2vec(act):
    vec =[0] * env.action_space.n
    vec[act] = 1
    return vec


def initial_population(number_games, render):
    training_data = []
    scores = []
    accepted_scores = []
    print('playing {} games (render = {})'.format(number_games, render))
    for game in xrange(number_games):
        score = 0
        game_memory = []
        prev_obs = []
        game_data = []
        for t in xrange(goal_steps):
            if render:
                env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])
            prev_obs = obs
            score += reward
            if done or info['ale.lives'] < 3:
                break
        score = t
        print('game = {}, score = {}'.format(game, score))
        if score >= score_requirement:
            print('good game')
            accepted_scores.append(score)
            for data in game_memory:
                out = act2vec(data[1])
                training_data.append([data[0], out])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    #np.save('training.npy', training_data_save)
    print('Average accepted score {}'.format(mean(accepted_scores)))
    print('Median accepted score'.format(median(accepted_scores)))
    print('Successful Games {} out of {}'.format(len(accepted_scores), number_games))
    print(Counter(accepted_scores))

    return training_data

def gen_train(training_file, render):
    gen_train = True
    if gen_train:
        trn = initial_population(render)
        np.save(training_file, np.array(trn))
        print('training written to {}'.format(training_file))
    else:
        with open ('outfile', 'rb') as fp:
            training_file = pickle.load(fp)
        print('training read from to {}'.format(training_file))


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--training", dest="training",
                      help="pre-computed training data")
    parser.add_option("-n", "--number_games", dest="number_games", default=20,
                      help="number of games", type="int")
    parser.add_option("-r", "--render",
                      action="store_false", dest="render", default=True,
                      help="render game board")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")
    (options, args) = parser.parse_args()

    training = options.training
    if training:
        print('using precomputed training data at {}'.format(training))
    else:
        print('training from scratch')
        t_start = time.time()
        trn = initial_population(options.number_games, options.render)
        t_end = time.time()
        print('generation time = {}'.format(t_end-t_start))
        training_file = 'data/si-training-gen.dat'
        t_start = time.time()
        np.save(training_file, np.array(trn))
        t_end = time.time()
        print('saving time = {}'.format(t_end-t_start))
        print('training save to {}'.format(training_file))


from __future__ import division

import gym
import tflearn
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from statistics import mean, median
from tflearn.layers.estimator import regression
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 10000

env._max_episode_steps = goal_steps

def ran_game():
    for ep in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_obs = []
        for _ in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])
            prev_obs = obs
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    out = [0, 1]
                else:
                    out = [1, 0]
                training_data.append([data[0], out])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('training.npy', training_data_save)
    print('Average accepted score', mean(accepted_scores))
    print('Median accepted score', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def make_nn(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='cart-pole')
    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    if not model:
        model = make_nn(input_size=len(X[0]))
    model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')
    return model

do_all = False
if do_all:
    training_data = initial_population()
    model = train_model(training_data)
    model.save('cart-pole.model')
else:
    model = make_nn(input_size=4)
    model.load('models/cart-pole-save.model')

scores = []
choices = []

for game in range(10):
    score = 0;
    game_memory = []
    prev_obs = []
    env.reset()
    for step in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            obs = prev_obs.reshape(-1, len(prev_obs), 1)
            inp = model.predict(obs)
            action = np.argmax(model.predict(obs)[0])
        choices.append(action)

        new_obs, reward, done, info = env.step(action)
        prev_obs = new_obs
        game_memory.append([new_obs, action])
        score += reward
        if done:
            print(step, score)
            break
    scores.append(score)

print('Average score {}'.format(mean(scores)))
print('Choice 1: {}, Choice {}'.format(choices.count(1)/len(choices),
                                       choices.count(0)/len(choices)))
print(Counter(scores))
#ran_game()






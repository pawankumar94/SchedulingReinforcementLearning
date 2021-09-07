import random
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
#import imageio
import os
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
# Todo : Check for epsilon update
# Todo: Include high kernel Size
# Todo: Train with Less Episodes to check results


import numpy as np


class Agent(object):
    def __init__(self, state_shape, env, optimizer, capacity):
        self.action_size = env.action_space.n
        self.optim = optimizer
        # self.image_shape = image_shape
        self.state_shape = state_shape
        self.n_rows, self.n_cols = self.state_shape
        # self.n_cols = self.state_shape[1]
        self.env = env
        self.epsilon_decay = 0.001
        self.experience_replay = deque(maxlen=capacity)
        self.gamma = 0.6
        self.epsilon = 0.1
        self.epsilon_min = 0.001
        self.q_network = self.build_compile_model()
        self.target_network = self.build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def build_compile_model(self):
        model = Sequential()
        model.add(Conv2D(32, 1, padding="valid", activation="relu",
                         input_shape=(self.n_rows, self.n_cols, 1)))
        model.add(Conv2D(64, 1, padding="valid", activation="relu",
                         input_shape=(self.n_rows, self.n_cols, 1)))
        model.add(Conv2D(64, 1, padding="valid", activation="relu",
                         input_shape=(self.n_rows, self.n_cols, 1)))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.action_size))
        huber = Huber()
        model.compile(loss=huber,
                      optimizer=self.optim,
                      metrics=["accuracy"])
        # print("Summary of the Model{}".format(model.summary()))
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        state = np.expand_dims(np.asarray(state).astype(np.float64), axis=2)
        state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(np.asarray(state).astype(np.float64), axis=2)
            state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
            next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=2)
            next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)
            target = self.q_network.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)
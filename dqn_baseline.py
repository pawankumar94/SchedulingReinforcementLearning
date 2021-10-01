import random
from collections import deque
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Activation, MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
from keras.callbacks import TensorBoard
import time
import tensorflow as tf
import numpy as np
from modifiedTensorBoard import  ModifiedTensorBoard
from config import *
Model_Name ="Test_Model"

class DQN:

    def __init__(self, input_shape, num_actions):
        self.OBSERVATION_SPACE_VALUES = input_shape
        self.ACTION_SPACE_SIZE = num_actions
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{Model_Name}-{int(time.time())}")
        self.memory = deque(maxlen=DRL_CFG["replay_size"])
        self.target_update_counter = 0

    def update_memory_replay(self, transition):
        self.memory.append(transition)

    def get_q_values(self, state):
       # state = np.moveaxis(state, 0, -1)
        state = state.reshape(-1, *state.shape)
        output = self.model.predict(state)[0]
        return output

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Train the Main network every Step
    def train(self, step, terminal):
        # start Training only if certain number of instances are collected
        if len(self.memory) < DRL_CFG["min_mem_size"]:
            return

        minibatch = random.sample(self.memory, DRL_CFG["BATCH_SIZE"])

        current_states = [transition[0] for transition in minibatch]  # we take the current states

        current_qs_list = self.model.predict(current_states, 32)  # provides the q values for all actions

        # We use the Target Model to predict the future values
        next_current_states = [transition[3] for transition in minibatch]  # we take the next State here
        future_qs_list = self.target_model.predict(next_current_states, 32)

        x = []
        y = []
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:  # if the episode is not over
                max_future_q = np.max(future_qs_list[index])
                print("For Index -->", index, "The Max Selected Value is --->", max_future_q)
                new_q = reward + DRL_CFG["GAMMA"] * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]  # Predicition of the Current Model
            current_qs[action] = new_q  # Update the Q value for the selected action

            x.append(current_state)
            y.append(current_qs)
        # Performing the Training here
        self.model.fit(x, y, batch_size=DRL_CFG["BATCH_SIZE"], callbacks=[self.tensorboard] if terminal else None)

        if terminal:
            self.target_update_counter += 1  # counter to initialize the target model

        if self.targe_update_counter > DRL_CFG["Update_target_every"]:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

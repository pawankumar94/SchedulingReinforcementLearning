from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
import random
class DqnAgent:
    def __init__(self, state_size, action_size):
        self.actions = action_size
        self.state_size = state_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_prob = 1.0
        self.exploration_decay = 0.005
        self.exploration_min = 0.001
        self.batch_size = 32
        self.max_mem_buffer = 100
        self.memory = deque(maxlen=self.max_mem_buffer)
        self.model = self.build_model()

    def compute_actions(self, current_state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(range(self.actions))
        q_values = self.model.predict(current_state)
        return np.argmax(q_values)

    # we update the epsilon value when the episode ends
    def update_exploration_rate(self):
        self.exploration_prob = self.exploration_prob - self.exploration_decay
        self.exploration_prob = max(self.exploration_prob, self.exploration_min)
        print("The updated Exploration Value ->", self.exploration_prob)

    # updated via each time steps
    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory.append({
            "current_state":current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done})
        if len(self.memory)> self.max_mem_buffer:
            self.memory.pop(0)

    def build_model(self):
        model = Sequential()
        optimizer = Adam(learning_rate=0.01)
        model.add(Conv2D(32, kernel_size=8, strides=4, padding="same", activation="relu",
                         input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu",
                         input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                         input_shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(8))
        huber = Huber()
        model.compile(loss=huber,
                      optimizer=optimizer,
                      metrics=["accuracy"],
                      )
        print("Summary of the Model{}".format(model.summary()))
        return model

    def train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for experience in minibatch:
            # we compute the q value of state St
            state = experience["current_state"]
            state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
            q_current_state = self.model.predict(state)
            q_target = experience["reward"]
            if not experience["done"]:
                next_state = experience["next_state"]
                next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)
                q_target = q_target + self.gamma*np.max(self.model.predict(next_state))
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(np.expand_dims(experience["current_state"],0), q_current_state, verbose=0)
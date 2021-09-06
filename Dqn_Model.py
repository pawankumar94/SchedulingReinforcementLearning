import torch
import math, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from config import *
np.random.seed(GYM_ENV_CFG['SEED'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, \
                                                                     batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, env):
        super(DQN, self).__init__()
        # The Input to the Model should be the Flattened Matrix
        self.layers = nn.Sequential(nn.Linear(num_inputs, 725),
                                    nn.ReLU(),
                                    nn.Linear(725,350),
                                    nn.ReLU(),
                                    nn.Linear(350, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_actions)
                                    )
        self.env = env

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state.T).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    @staticmethod
    def compute_td_loss(batch_size, replay_buffer, model, optimizer):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.flatten()  # here we need to Flatten the State
        next_state = next_state.flatten()
        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.tensor(action))
        reward = Variable(torch.tensor(reward))
        done = Variable(torch.tensor(done))
        q_values = model(state)
        next_q_values = model(next_state)
        q_value = q_values.gather(0, action.unsqueeze(0)[0]).squeeze(0)
        next_q_value= next_q_values.max(0)[0]
        expected_q_value = reward + DRL_CFG['GAMMA'] * next_q_value
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()   # TD Error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
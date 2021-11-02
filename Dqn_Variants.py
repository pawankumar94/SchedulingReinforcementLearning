import torch
from data_preprocess import *
from config import *
import torch
import torch.nn as nn
import random
import torch.autograd as autograd
import torch.optim as optim
import operator
import numpy as np
import pandas as pd
from collections import deque
import operator
from torch.autograd import Variable
import gym
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DuellingQNetworks(nn.Module):
    def __init__(self, env):
        super(DuellingQNetworks, self).__init__()
        self.env = env
        self.state_dim = 48
        self.num_actions = env.action_space.n
        self.h1 = self.state_dim * 2
        self.h2 = 128

        self.feature = nn.Sequential(
            nn.Linear(self.state_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, self.num_actions)
                                    )

        self.advantage = nn.Sequential(
            nn.Linear(self.state_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, self.num_actions)
                                      )

        self.value = nn.Sequential(
            nn.Linear(self.state_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, 1)
                                  )

    def forward(self, state):
        state = torch.flatten(state, 1).to(device)
        advantage = self.advantage(state)
        value = self.value(state)
        return (value + advantage) - advantage.mean()

    def select_action(self, state, epsilon):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = self.env.get_valid_action_mask()

        machines_from_mask = np.where(np.array(mask) == 1)[0]
        sample = random.random()

        if sample > epsilon:
            with torch.no_grad():

                state = Variable(torch.FloatTensor(state).unsqueeze(0))
                q_values = self.forward(state)
                q_list = q_values.tolist()[0]
                q_valid_dict = {v_action: q_list[v_action] for v_action in machines_from_mask}
                action = max(q_valid_dict.items(), key=operator.itemgetter(1))[0]

        else:

            action = int(np.random.choice(machines_from_mask, 1))

        return torch.tensor(action, device=device, dtype=torch.long)


class DQN(nn.Module):

    def __init__(self, env):
        super(DQN, self).__init__()
        self.env = env
        self.state_dim = 48
        self.num_actions = env.action_space.n
        self.h1 = self.state_dim * 2
        self.h2 = 128

        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, self.num_actions)
        )

    def forward(self, x):
        x = torch.flatten(x, 1).to(device)  # to process batch
        q_vals = self.layers(x)
        return q_vals

    def select_action(self, state, epsilon):

        mask = self.env.get_valid_action_mask()
        # mask = get_action_mask(self.env)
        # device = "cpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        machines_from_mask = np.where(np.array(mask) == 1)[0]
        sample = random.random()

        if sample > epsilon:
            with torch.no_grad():

                state = Variable(torch.FloatTensor(state).unsqueeze(0))
                q_values = self.forward(state)
                q_list = q_values.tolist()[0]
                q_valid_dict = {v_action: q_list[v_action] for v_action in machines_from_mask}
                action = max(q_valid_dict.items(), key=operator.itemgetter(1))[0]

        else:

            action = int(np.random.choice(machines_from_mask, 1))

        return torch.tensor(action, device=device, dtype=torch.long)


class Replay_Buffer(nn.Module):
    def __init__(self, capacity):
        super(Replay_Buffer, self).__init__()
        self.replay = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.replay, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def len(self):
        return len(self.replay)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        w1 = (total * probs[indices])

        weights = np.power(w1, (-beta))  # w1 ** -beta
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def len(self):
        return len(self.buffer)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def soft_update(current_model, target_model):
    """Soft update model parameters.
           θ_target = τ*θ_local + (1 - τ)*θ_target
           Params
           =======
               local model (PyTorch model): weights will be copied from
               target model (PyTorch model): weights will be copied to
               tau (float): interpolation parameter
           """
    tau = 0.001
    for target_param, local_param in zip(target_model.parameters(),
                                         current_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


def decay_epsilon(epsilon):
    epsilon *= DRL_CFG['epsilon_decay']
    epsilon = max(epsilon, DRL_CFG['epsilon_final'] )
    return epsilon


def compute_loss(replay_buffer, current_model, target_model, batch_size, beta=None):
    gamma = DRL_CFG['GAMMA']

    if DRL_CFG['Optimizer'] == "RMSPROP":
        optimizer = optim.RMSprop(current_model.parameters(), lr=DRL_CFG['LR'], eps=0.001, alpha=0.95)
    else:
        optimizer = optim.Adam(current_model.parameters(), DRL_CFG['LR'])

    if DRL_CFG['BUFFER_TYPE'] == 'NORMAL':

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    else:
        state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)
        weights = Variable(torch.FloatTensor(weights))
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state = Variable(torch.FloatTensor(np.float32(state))).to(device)
    next_state = Variable(torch.FloatTensor(np.float32(next_state))).to(device)
    action = Variable(torch.LongTensor(action)).to(device)
    reward = Variable(torch.FloatTensor(reward)).to(device)
    done = Variable(torch.FloatTensor(done)).to(device)

    q_values = current_model(state)
    next_q_values = current_model(next_state)

    if DRL_CFG['MODEL_ARCH'] == 'DQN':
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

    elif DRL_CFG['MODEL_ARCH'] == 'DDQN':
        next_q_state_values = target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

    if DRL_CFG['BUFFER_TYPE'] == 'PER':
        if DRL_CFG['Loss'] == "TDERROR":
            loss = ((q_value.to(device) - expected_q_value.detach().to(device)).pow(2) * weights.to(device)).to(device)
            prios = loss + 1e5
            loss = loss.mean().to(device)
            optimizer.zero_grad()
            loss.backward()
            # clipping the Gradients
            clip_val = DRL_CFG['Gradien_Clip_Val']
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), clip_val)
            #  for param in current_model.parameters():
            #   param.grad.data.clamp_(-1, 1)
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            optimizer.step()

        else:
            criteria = nn.SmoothL1Loss()
            loss = criteria(q_value.to(device), expected_q_value.detach().to(device)) ** weights.to(device)
            prios = loss + 1e5
            loss = loss.mean().to(device)
            optimizer.zero_grad()
            loss.backward()
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            optimizer.step()
    else:
        if DRL_CFG['Loss'] == "TDERROR":
            loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            clip_val = DRL_CFG['Gradien_Clip_Val']
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), clip_val)
            # for param in current_model.parameters():
            # param.grad.data.clamp_(-1, 1)
            optimizer.step()
        else:
            loss = nn.SmoothL1Loss(q_value.to(device), Variable(expected_q_value.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss

def save_model(q_model, target_model, path):
    name = "q_model.pth"
    target_name = "target_model.pth"
    torch.save(q_model, path+name)
    torch.save(target_model,path+target_name)


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
np.random.seed(GYM_ENV_CFG['SEED'])
from gym_custom.envs.custom_env_1 import  customEnv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_data_set():

    # Google Dataset has 4 Categories of Tasks with the following values as the Maximum request made by each Task
    category_of_jobs = {'Type0': {"cpu_req": 0.04, "mem_req": 0.038}, 'Type1': {"cpu_req": 0.065, "mem_req": 0.07},
                        'Type2': {"cpu_req": 0.03, "mem_req": 0.03}, 'Type3': {"cpu_req": 0.1, "mem_req": 0.12},
                        }
    # Freuqency of Occurence for Each category of Task
    percentage_jobs_each_category = [0.38, 0.32, 0.29, 0.01]
    np.random.seed(GLOBAL_CFG["SEED"])
    sample_dict = {"cpu_req": [],
                   "mem_req": [],
                   "class": [],
                   }

    for _ in range(1_00_000):
        probs = percentage_jobs_each_category
        for dim in GLOBAL_CFG["features_to_include"]:
            type_0 = np.random.uniform(low=0, high=category_of_jobs["Type0"][dim], size=1)[0]
            type_1 = np.random.uniform(low=0, high=category_of_jobs["Type1"][dim], size=1)[0]
            type_2 = np.random.uniform(low=0, high=category_of_jobs["Type2"][dim], size=1)[0]
            type_3 = np.random.uniform(low=0, high=category_of_jobs["Type3"][dim], size=1)[0]
            a = [type_0, type_1, type_2, type_3]
            sample = random.choices(population=list(enumerate(a)), weights=probs, k=1)
            #    sample = np.random.choice(a, size=1, p=probs)[0]
            class_type, value = sample[0]
            sample_dict[dim].append(value)
            if dim == "cpu_req":
                sample_dict["class"].append(class_type)

    synthetic_df = pd.DataFrame(sample_dict)
    synthetic_df['cpu_rate'] = synthetic_df['cpu_req'] * 0.5
    synthetic_df['can_mem_usg'] = synthetic_df['mem_req'] * 0.5
    synthetic_df.drop(['class'], axis=1, inplace=True)
    data = np.zeros(synthetic_df.shape[0])
    synthetic_df.insert(0, 'Placed', data)
    synthetic_df.insert(1, "Task_requested_time", data)
    synthetic_df.insert(len(synthetic_df.columns), "Done", data)

    train_data, attr_idx, state_indices = preprocess_data(synthetic_df, 1000)
    subset_dataset = {}
    subset_task_duration = {}
    train_data = train_data.tolist()
    for i in range(GLOBAL_CFG['Max_No_of_Jobs']):
        subset_dataset[i] = np.asarray(random.sample(train_data, random.randint(1, GLOBAL_CFG['Max_No_of_Task'])))
        subset_task_duration[i] = generate_duration(subset_dataset, key=i,
                                                    length_duration=GLOBAL_CFG["TASK_DURS_MEDIUM"])

    epi_dur = {}
    for key in subset_dataset.keys():
        epi_dur[key] = (subset_dataset[key].shape[0])

    max_epi_dur = max(epi_dur, key=epi_dur.get)
    ### Creating 100 same episodes
    dict_with_one_epi = {}
    duration_for_one_epi = {}
    for epi in range(100):
        dict_with_one_epi[epi] = subset_dataset[max_epi_dur]
        duration_for_one_epi[epi] = subset_task_duration[max_epi_dur]
    env = gym.make('custom-v0',
                   train_data=dict_with_one_epi,
                   task_duration=duration_for_one_epi,
                   state_idx=state_indices,
                   attr_idx=attr_idx)
    return env

class DQN(nn.Module):

    def __init__(self, env):
        super(DQN, self).__init__()
        self.env = env
        self.state_dim = 48
        self.num_actions = env.action_space.n
        self.h1 = self.state_dim *2
        self.h2 = 128

        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, self.num_actions)
                                   )

    def forward(self, x):
        x = torch.flatten(x, 1) # to process batch
        q_vals = self.layers(x)
        return q_vals

    def select_action(self, state, epsilon):

        mask = self.env.get_valid_action_mask()
        machines_from_mask = np.where(np.array(mask) == 1)[0]
        sample = random.random()

        if sample > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_values = self.forward(state)
            q_list = q_values.tolist()[0]
            q_valid_dict = {v_action: q_list[v_action] for v_action in machines_from_mask}
            action = max(q_valid_dict.items(), key=operator.itemgetter(1))[0]
        else:
            action = int(np.random.choice(machines_from_mask, 1))

        return action

class Replay_Buffer(nn.Module):
    def __init__(self, capacity):
        super(Replay_Buffer, self).__init__()
        self.replay = deque(maxlen= capacity)

    def push(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.replay, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def len(self):
        return len(self.replay)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def decay_epsilon(epsilon):
    epsilon *= DRL_CFG['epsilon_decay']
    epsilon = max(epsilon,DRL_CFG['epsilon_final'] )
    return epsilon

def compute_td_loss(replay_buffer, current_model, target_model, batch_size, beta = None):
    gamma = DRL_CFG['GAMMA']
    optimizer = optim.Adam(current_model.parameters())

    if DRL_CFG['BUFFER'] == 'Normal':
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    else:
        state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)
        weights = Variable(torch.FloatTensor(weights))

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

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

    if DRL_CFG['BUFFER'] == 'PER':
        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e5
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        optimizer.step()

    else:
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




    return loss


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
       # state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)

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
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

from matplotlib.pyplot import step
from config import *
from data_preprocess import *
import random
import gym_custom
import gym
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
res_dist_dict = {}
for res in ['cpu_req', 'mem_req']:
    res_dist_dict[res] = GYM_ENV_CFG[res]
df = data_gen(1_000, res_dist_dict, ratio=0.1)
train_data, attr_idx, state_indices = preprocess_data(df, 2000)
subset_dataset = {}
subset_task_duration = {}
train_data = train_data.tolist()
for i in range(GLOBAL_CFG['Max_No_of_Jobs']):
    subset_dataset[i] = np.asarray(random.sample(train_data, np.random.choice\
        (GLOBAL_CFG['Max_No_of_Task'])))
    subset_task_duration[i] = generate_duration(subset_dataset, key=i)

env = gym.make('custom-v0',
               train_data=subset_dataset,
               task_duration=subset_task_duration,
               state_idx=state_indices,
               attr_idx=attr_idx
              )
total_steps=0
replay_mem = []
for episode in range(1):
        state = env.reset()
        done = False
        current_epi_reward = 0
        while not done:
            total_steps += 1
            #action = np.random.choice(GYM_ENV_CFG['NB_NODES'] +1)
            pos = {0: 4, 8: 4, 6: 3}
            action = random.choice([x for x in pos for y in range(pos[x])])
            print("Action:", action)
            print("StepNumber:", env.i)
            next_state, reward, done, _ = env.step(action)
            print("reward:", reward)
            replay_mem.append([state, next_state, reward, done])
            state = next_state
            if done:
                print("episodeReward", reward)

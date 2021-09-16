from Dqn_Agent import DqnAgent
from data_preprocess import *
from config import *
import numpy as np
import pandas as pd
import gym
import gym_custom
import random

np.random.seed(GYM_ENV_CFG['SEED'])
#script_dir = os.path.dirname(__file__)
#results_dir = os.path.join(script_dir, 'Results/')
res_dist_dict = {}
for res in ['cpu_req', 'mem_req']:
    res_dist_dict[res] = GYM_ENV_CFG[res]
df = data_gen(1_000, res_dist_dict, ratio=0.1)
train_data, attr_idx, state_indices = preprocess_data(df, 2000)
subset_dataset = {}
subset_task_duration = {}
train_data = train_data.tolist()
for i in range(GLOBAL_CFG['Max_No_of_Jobs']):
    subset_dataset[i] = np.asarray(random.sample(train_data,\
                                                 random.randint(1, GLOBAL_CFG['Max_No_of_Task'])))
    subset_task_duration[i] = generate_duration(subset_dataset, key=i,\
                                                length_duration=GLOBAL_CFG['TASK_DURS_MEDIUM'])

env = gym.make('custom-v0',
               train_data=subset_dataset,
               task_duration=subset_task_duration,
               state_idx=state_indices,
               attr_idx=attr_idx
              )
state_size = env.reset().shape
no_actions = env.action_space.n
_obj = DqnAgent(state_size, no_actions)
number_episodes = GLOBAL_CFG["Max_No_of_Jobs"]
total_steps = 0
batch_size = 32
episode_reward = {}
per_episode_reward = {}
for e in range(5):
  current_state = env.reset()
  steps_in_current_epi = len(env.all_episodes_duration[e])
  print(current_state.shape)
  per_episode_reward[e] = []
  for step in range(steps_in_current_epi):
    total_steps +=1
    action = _obj.compute_actions(current_state)
    next_state, reward, done, _ = env.step(action)
    per_episode_reward[e].append(reward)
    _obj.store_episode(current_state, action, reward, next_state, done)
    if done:
      _obj.update_exploration_rate()
      episode_reward[e] = reward
      print("End of the Episode")
  # we train our Model Here
  if total_steps > batch_size:
      _obj.train()


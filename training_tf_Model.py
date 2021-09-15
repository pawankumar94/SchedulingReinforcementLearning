from config import *
from data_preprocess import *
import tensorflow as tf
import numpy as np
import gym , gym_custom
import random
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from tf_model import Agent
import os
import matplotlib.pyplot as plt
import imageio
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
    subset_task_duration[i] = generate_duration(subset_dataset, key=i)

env = gym.make('custom-v0',
               train_data=subset_dataset,
               task_duration=subset_task_duration,
               state_idx=state_indices,
               attr_idx=attr_idx
              )


def gen_plot(state, timestep, path_to_dir):
    cpu_usgages = state[0][0][4:4 + 8]
    mem_usages = state[0][0][4 + 8:4 + 8 * 2]
    fig = plt.figure(figsize=(10, 5))
    n = GYM_ENV_CFG['NB_NODES']
    r = np.arange(n)
    width = 0.25
    plt.bar(r, cpu_usgages, color='g',
            width=width, edgecolor='black',
            label='Cpu_usage')
    plt.bar(r + width, mem_usages, color='r',
            width=width, edgecolor='black',
            label='Memory Usage')

    plt.xlabel("No Of Machine")
    plt.ylabel("Usage Per Machine")
    plt.title("TimeStep" + str(1))

    # plt.grid(linestyle='--')
    plt.xticks(r + width / 2, ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])
    plt.legend()
    plt.show()
    name = str(timestep) + ".jpg"
    fig.savefig(path_to_dir + name, bbox_inches='tight', dpi=150)


def gen_gif(path_dir):
    png_dir = 'path_dir'
    images = []
    for file_name in sorted(os.listdir(path_dir)):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(path_dir, file_name)
            images.append(imageio.imread(file_path))
    filename = "Movie1.gif"
    imageio.mimsave(path_dir+filename, images)

optimizer = Adam(learning_rate= 0.01)
state = env.reset()
state_shape = state.shape
agent = Agent(state_shape=state_shape, env = env, optimizer= optimizer, capacity= 1000)
batch_size = 32
no_episodes = len(env.all_episodes_duration)
os.makedirs("Output")

for episode in tqdm(range(0, no_episodes)):
    state = env.reset()
    reward = 0
    done = False
    timestep_current_epi = 0
    reward_list = []
    action_list = []
    state_list = []
    dir_name = str(episode)
    for timestep in range(len(env.all_episodes_duration[episode])):
        timestep_current_epi += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        state = next_state
        print("Episode No", episode, "TimeStep:", timestep_current_epi)
        if done:
            print("End oF Episode: {}".format(episode), "Total No of Steps Taken {}".format(timestep_current_epi))
            print("Episode Reward:", reward)
            agent.align_target_model()
            break

        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)



from config import *
from data_preprocess import *
import random
import gym_custom
import gym
import matplotlib.pyplot as plt
import os
import time
from dqn_baseline import DQN
from tqdm import tqdm

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

# Create models folder only if you want to save the model
'''if not os.path.isdir('models'):
    os.makedirs('models')'''

input = np.moveaxis(env.reset(), 0, -1)  # here we move the channels to the last axis
input_shape = input.shape
action_space = env.action_space.n
agent = DQN(input_shape, action_space)
epsilon = DRL_CFG["epsilon_start"]
epi_rewards = []
aggregrate_stats_every = 50
show_preview = True
ep_rewards = []
for episode in tqdm(range(0, GLOBAL_CFG["Max_No_of_Jobs"]+1), ascii= True, unit="episode"):
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 0
    current_state = env.reset()  # shape (1,rows,cols)
    done = False
    while not done:
        if random.random() > epsilon:
            action = np.argmax(agent.get_q_values(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)

        next_state, reward, done, _ = env.step(action)  # next state shape same
        episode_reward += reward
        # Change the axis

        #current_state = np.moveaxis(current_state, 0, -1)
        #next_state = np.moveaxis(next_state, 0, -1)

        agent.update_memory_replay((np.moveaxis(current_state, 0, -1), action, reward, np.moveaxis(next_state, 0, -1), done))
        agent.train(terminal=done, step=step)
        current_state = next_state
        step += 1
        if done:
            print("end of episode", env.episode_no, "length:", len( env.all_episodes_duration[env.episode_no]))


        # epi reward
    epi_rewards.append(episode_reward)
    '''if not episode % aggregrate_stats_every or episode == 1:
        average_reward = sum(ep_rewards[-aggregrate_stats_every:]) / len(ep_rewards[-aggregrate_stats_every:])
        min_reward = min(ep_rewards[-aggregrate_stats_every:])
        max_reward = max(ep_rewards[-aggregrate_stats_every:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)'''
    if epsilon > DRL_CFG['epsilon_final']:
        epsilon *= DRL_CFG["epsilon_decay"]
        epsilon = max(DRL_CFG["epsilon_final"], epsilon)




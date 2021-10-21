from config import *
from data_preprocess import *
import random
import gym_custom
import gym
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

np.random.seed(GYM_ENV_CFG['SEED'])
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

# Second alternate if machines are overbooked we give high penalty and return the copy of  \
# state and does not increment counter
# enabling that the machine cannot place on Machines with > 100
'''def masking_machine():
    percentage_machine_used = env.calculate_percent_machine()
    list_of_feasible_machine =[]
    for machine in percentage_machine_used:
        cpu_usage = percentage_machine_used[machine][0]
        mem_usage = percentage_machine_used[machine][1]
        if not (cpu_usage >= 100) | (mem_usage >= 100):
            list_of_feasible_machine.append(machine)
    list_of_feasible_machine = [z+1 for z in list_of_feasible_machine]
    list_of_feasible_machine.insert(0, 0)
    return list_of_feasible_machine'''

total_steps= 0
replay_mem = []
for episode in range(GLOBAL_CFG['Max_No_of_Jobs']):
        state = env.reset()
        done = False
        current_epi_reward = 0
        dir_name = str(episode)+'/'
        path = Path(results_dir+dir_name)
        path.mkdir(parents=True, exist_ok=True)
        while not done:
            total_steps += 1
            #pos = {0: 4, 8: 4, 6: 3}
            #action = random.choice([x for x in pos for y in range(pos[x])])
            #feasible_actions = masking_machine()
            #action = random.choice(feasible_actions)
            #action = np.random.choice(GYM_ENV_CFG['NB_NODES']+1) ## model.predict()
            action_mask = env.get_valid_action_mask()
            action = 3
            print("Episode_Number:", env.episode_no)
            print("StepNumber:", env.i)
            next_state, reward, done, info = env.step(action)
            print("Information:", info)
            print("reward:", reward)
            replay_mem.append({"state":state, "next_state":next_state})
            state = next_state

            env.gen_plot(path_to_dir=path)
            if done:
                print("episodeReward", reward)
                print('\n', "***********")
               # print(np.shape(state))
                env.make_gif(path=path)
                time.sleep(5)

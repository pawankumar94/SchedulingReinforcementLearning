from matplotlib.pyplot import step
from config import *
from data_preprocess import *
import random
import custom_env_1
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

_obj = custom_env_1.customEnv(train_data=subset_dataset,
                              attr_idx=attr_idx,
                              task_duration=subset_task_duration,
                              state_idx= state_indices
                              )

print(_obj.start_idx)

state = _obj.reset()
for episode in range(GLOBAL_CFG['Max_No_of_Jobs']):
    print("Beginning of New Episode:", _obj.episode_no)
    done = False
    counter = 0
    reward_list, steps_list, action_list = [], [], []
    while not done:
        action = np.random.choice(9)
        number_of_step = _obj.i
        print("Step Number", number_of_step)
        print("Action Taken", action)
        next_state, reward, done, _ = _obj.step(action)
        print("Reward", reward)
        reward_list.append(reward)
        steps_list.append(number_of_step)
        action_list.append(action)
        if number_of_step % 1 == 0:
            counter += 1
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_figheight(8)
            fig.set_figwidth(13)
            ax1.plot(steps_list, reward_list)
            ax1.set_title("Steps to Rewards")
            ax2.plot(steps_list, action_list)
            ax2.set_title("Steps to actions")
            sample_file_name = "result"+"_"+str(counter)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.savefig(results_dir + sample_file_name)
            counter += 1
        if done:
            print(" Episode Reward", reward)
            break
    break
import copy
import random
import gym
import numpy as np
from config import *
from rewards import *
from gym import spaces
import os
import imageio
import matplotlib.pyplot as plt
import glob
from PIL import Image
import natsort
from data_preprocess import *
np.random.seed(GYM_ENV_CFG['SEED'])

class customEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 train_data,
                 task_duration,
                 state_idx,
                 attr_idx,
                 random_initialize= False):
        super(customEnv, self).__init__()
        self.train_data = train_data  # loading the dataset
        self.all_episodes_duration = task_duration
        # val:2
        self.state_idx = state_idx
        self.attr_idx = attr_idx  # cpu:2 and memory:3
        self.nb_dim = len(self.state_idx)
        # self.j = 0  # steps including wait task
        self.random_initialize = random_initialize  # Flag to determine if some machines should\
        self.nb_w_nodes = GYM_ENV_CFG["NB_NODES"]
        self.reset()
        # Used for Over-util Reward
        self.overshoot_counter = 0
        self.machine_status = {}

    def no_more_steps(self):
        return self.i == self.max_steps_current_epi-1

    def get_task_usages(self):
        # we extract the cpu and memory usages of the tasks
        cpu_usage_col = self.attr_idx['cpu_rate']
        mem_usg_col = self.attr_idx['can_mem_usg']
        # Cpu Usage Mde by current Running Task
        real_cpu_usg = self.train_data_epi[self.i][cpu_usage_col]
        # Mem Usage made by current running task
        real_mem_usg = self.train_data_epi[self.i][mem_usg_col]
        return real_cpu_usg, real_mem_usg

    def move_to_next_task(self):
        self.state[:self.nb_dim] = self.train_data_epi[self.i][self.nb_dim:self.nb_dim * 2]

    def update_state_machine_limits(self):
        # We will update the machine capacities in the State
        cpu_limits = []
        mem_limits = []
        for machine in self.machine_capacity:
            # Changed machine capacity at each step
            cpu_value = self.machine_capacity[machine][0]
            mem_value = self.machine_capacity[machine][1]
            cpu_limits.append(cpu_value)
            mem_limits.append(mem_value)
        # Updating the machine Limits in State
        self.state[self.nb_dim + (self.nb_w_nodes * 2):self.nb_dim + (self.nb_w_nodes * 3)] = \
            cpu_limits
        self.state[self.nb_dim + (self.nb_w_nodes * 3):self.nb_dim + (self.nb_w_nodes * 4)] = mem_limits

    def change_in_machine_capacity(self, action, usages, placed=False):
        if placed:
            capacity = self.machine_capacity[action]
            diff = [x - y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = diff
        else:
            capacity = self.machine_capacity[action]
            add = [x + y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = add
        return

    def update_state(self, tasks= None, remove = False):
        if remove:
            for key in tasks:
                machine_no, cpu_usage, mem_usage = list(self.memory[key].values())
                usages = [cpu_usage, mem_usage]
                self.state[(self.nb_dim + machine_no)-1] -= cpu_usage
                self.state[(self.nb_dim + (machine_no) + self.nb_w_nodes)-1] -= mem_usage
                self.change_in_machine_capacity(action=machine_no,usages=usages )
                self.update_state_machine_limits()
        else:
            for machine in self.machine_status:
                for item in self.machine_status[machine]:
                    machine_cpu_usg = item['cpu']
                    machine_mem_usg = item['mem']
                    # update the cpu usage of each machine
                    self.state[(self.nb_dim + machine)-1] += machine_cpu_usg  # Cpu Usage Col
                    # update the memory usage of each machine
                    self.state[(self.nb_dim + machine + self.nb_w_nodes)-1] += machine_mem_usg
                    self.update_state_machine_limits()
    def wait(self):
        wait_time = min(self.running_task.values())
        self.timer += wait_time
        self.wait_counter += 1
        tasks_ends_now = [k for k, v in self.running_task.items() if v == wait_time]
        self.update_state(tasks = tasks_ends_now, remove = True)
        [self.running_task.pop(key) for key in tasks_ends_now]

        for key, value in self.running_task.items():
            self.running_task[key] = value - wait_time

    def check_if_task_should_end(self):
        x = []
        for keys in self.running_task:
            time = self.running_task[keys]
            if time == self.timer:
                x.append(keys)
        return x

    def step(self, action):
        action = int(action)
        info = {}
        cpu_usage, mem_usage = self.get_task_usages()
        usages_for_machine = [cpu_usage, mem_usage]
        execution_time_task = self.epi_duration[self.i]
        cpu_limit, mem_limit = self.machine_capacity[action]
        task_ending_currently = self.check_if_task_should_end()

        if (cpu_limit < 0.001) or (mem_limit< 0.001):
            self.wait()
        elif len(task_ending_currently) != 0:
            self.update_state(remove=True, tasks=task_ending_currently)

        self.memory[self.i] = {"Machine_No": action,
                               "cpu_usage": cpu_usage,
                               "mem_usage": mem_usage,
                               }

        self.machine_status[action].append({
            "cpu": cpu_usage,
            "mem": mem_usage,
            "rem_time": execution_time_task,
            "task_id": self.i,
            "cpu_req": self.train_data[self.episode_no][self.i][self.attr_idx['cpu_req']],
            "mem_req": self.train_data[self.episode_no][self.i][self.attr_idx['mem_req']]
        })

        self.running_task[self.i] = execution_time_task
        self.change_in_machine_capacity(placed=True,action=action, usages=usages_for_machine)
        self.update_state()
        self.timer += self.time_to_process_each_task
        usage = list(self.state[self.nb_dim:self.nb_dim + (self.nb_w_nodes * 2)])

        if DRL_CFG['reward_type'] == 'under_util':
            self.reward = under_util_reward(usage)

        elif DRL_CFG['reward_type'] == 'simple':
            self.reward = get_intermediate_reward(action=action, usages=usage, updated_capacities=self.machine_capacity)

        elif DRL_CFG['reward_type'] == 'over_util':
            self.reward = over_util_reward()

        self.i += 1  # increment only when we place task
        percentage_used_machine = self.calculate_percent_machine()
        info["machine-Used-Percentage"] = percentage_used_machine
        self.move_to_next_task()
        self.sum_reward.append(self.reward)
        if self.no_more_steps():
            self.done = True
            #self.reward = new_episode_end_reward(wait_counter=self.wait_counter,
             #                                    cum_reward=self.sum_reward)
            max_run_task = max(self.running_task.values())
            self.timer+=max_run_task
            self.running_task = {}

            for machine in range(self.nb_w_nodes):
                self.machine_capacity[machine] = machine_limits(machine)

            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
            self.episode_no +=1

        return self.state, float(self.reward),self.done, info

    def reset(self):
        self.timer = 0
        self.episode_no = 0
        self.time_to_place_task = 10
        self.state_index = self.nb_dim + (self.nb_w_nodes * 4)
        self.done = False
        self.action_space = spaces.Discrete(self.nb_w_nodes)  #
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.state_index, 1),
                                            dtype=np.float64)

        self.state = np.zeros(self.state_index)
        self.i = 0

        self.wait_counter = 0
        self.total_wait_time_epi = 0
        self.reward = 0
        self.sum_reward = []

        self.machine_status={}

        self.machine_capacity = {}
       # self.machine_status = {}
        self.mask = []
        self.memory = {}

        for machine in range(self.nb_w_nodes):
            self.machine_capacity[machine] = machine_limits(machine)

        self.update_state_machine_limits()

        self.time_to_process_each_task = 10
        self.wait_time_current_epi = 0
        self.max_steps_current_epi = len(self.train_data[self.episode_no])

        self.train_data_epi  = self.train_data[self.episode_no]
        self.epi_duration = self.all_episodes_duration[self.episode_no]
        self.move_to_next_task()
        self.running_task = {}
        for idx in range(self.nb_w_nodes):
            self.machine_status[idx] = []

        return self.state

    # Utility functions
    def gen_plot(self, timestep=None, path_to_dir=None):
        state = self.state
        timestep = self.i
        percentage_used_machine = self.calculate_percent_machine()
        cpu_usages = []
        mem_usages = []
        for key in percentage_used_machine:
            cpu_usages.append(percentage_used_machine[key][0])
            mem_usages.append(percentage_used_machine[key][1])
        fig = plt.figure(figsize=(10, 5))
        n = GYM_ENV_CFG['NB_NODES']
        r = np.arange(n)
        width = 0.25
        plt.bar(r, cpu_usages, color='g',
                width=width, edgecolor='black',
                label='Cpu_usage')
        plt.bar(r + width, mem_usages, color='r',
                width=width, edgecolor='black',
                label='Memory Usage')

        plt.xlabel("No Of Machine")
        plt.ylabel("Usage Per Machine")
        plt.title("TimeStep" + str(timestep))
        # plt.grid(linestyle='--')
        plt.xticks(r + width / 2, ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'])
        plt.ylim(0, 100)
        plt.legend()
        #plt.show()
        name = str(timestep) + ".jpeg"
        fig.savefig(os.path.join(path_to_dir, name), bbox_inches='tight', dpi=150)
        plt.close()

    def make_gif(self,path):
        png_dir = path
        images = []
        files = os.listdir(png_dir)
        sorted_list = natsort.natsorted(files,reverse=False)
        for file_name in sorted_list:
            if file_name.endswith('.jpeg'):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))
        gif_name = "movie.gif"
        imageio.mimsave(os.path.join(path, gif_name), images, format='GIF', duration = 0.4)

    def get_metric(self):
        info = {}
        # Log the reward
        if self.no_more_steps(): #or self.termination_conditon_waiting():
            # we complete all the running Tasks
            percentage_used_machine = self.calculate_percent_machine()
            info["Final_Machines_Percentage_usage"] = percentage_used_machine
            percent_of_task_completed, total_no_of_tasks \
                , total_steps_including_waiting, total_steps_excluding_wait \
                = self.calculate_task_completed_epi()
            info["Percentage_Task_Completed"] = percent_of_task_completed
            info["Total_Task_Episode"] = total_no_of_tasks
            info["Steps_Including_Wait"] = total_steps_including_waiting
            info["Steps_Without_Wait"] = total_steps_excluding_wait
            info["Wait_steps_taken"] = total_steps_including_waiting - total_steps_excluding_wait
            info["Episode_End_Reward"] = self.reward
            info["Cumulative_Reward"] = self.cum_reward
        else:
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
            info["Step-Reward"] = self.reward
        return info

    def calculate_task_completed_epi(self):
        total_task_current_epi = len(self.train_data[self.episode_no])-1
        percent_of_task_completed = (self.i / total_task_current_epi)*100
        total_steps_including_waiting = self.j
        total_steps_excluding_wait = self.i
        output_info = [percent_of_task_completed, total_task_current_epi, total_steps_including_waiting, total_steps_excluding_wait]
        return output_info

    def calculate_percent_machine(self):
        percentage_per_machine = {}
        for machine in range(self.nb_w_nodes):  #8
            cpu_limit, memory_limit = self.machine_limits(machine)  # orignal limit of Machines
            changed_cpu_limit, changed_mem_limit = self.machine_capacity[machine]  # Changed Limit of the Machine
            percentage_cpu = ((cpu_limit - changed_cpu_limit) / cpu_limit) * 100  # percentage of Machine Used after placement5
            percentage_mem = ((memory_limit - changed_mem_limit) / memory_limit) * 100
            percentage_per_machine[machine] = [percentage_cpu, percentage_mem]
        return percentage_per_machine

    def get_valid_action_mask(self):
        x = np.ones(self.action_space.n)
        for element in self.machine_capacity:
            cpu_capacity = self.machine_capacity[element][0]
            mem_capacity = self.machine_capacity[element][1]
            if (cpu_capacity <= 0) or (mem_capacity <= 0):
                x[0] = 0
        return x

    def machine_limits(self, machine):
        machine_cpu_type = GYM_ENV_CFG['MC_CAP'][machine]
        machine_cpu_cap = GLOBAL_CFG['MC_CAP_VALUES'][machine_cpu_type]
        machine_mem_type = GYM_ENV_CFG['MM_CAP'][machine]
        machine_mem_cap = GLOBAL_CFG['MM_CAP_VALUES'][machine_mem_type]
        return [machine_cpu_cap, machine_mem_cap]


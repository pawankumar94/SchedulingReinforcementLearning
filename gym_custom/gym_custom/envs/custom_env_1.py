import copy
import random
import gym
import numpy as np
from config import *
from gym import spaces
import os
import imageio
import matplotlib.pyplot as plt
import glob
from PIL import Image
import natsort
np.random.seed(GYM_ENV_CFG['SEED'])
from rewards import *

# A task aquires 50 % of requests and does not increase at each timestep

class customEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 train_data,
                 task_duration,
                 state_idx,
                 attr_idx,
                 start_idx=0,
                 random_initialize=False,
                 ):
        super(customEnv, self).__init__()
        #self.random_initialize = random_initialize

        # Condition If we want to process one task at a time
        self.one_task = GYM_ENV_CFG['One-Task']
        # loading the dataset
        self.train_data = train_data
        # Duration for all episode tasks
        self.all_episodes_duration = task_duration
        # [cpu_req, mem_req]
        self.state_idx = state_idx
        self.episode_no = 0
        # cpu:0 and memory:1
        self.attr_idx = attr_idx
        #self.start_idx = start_idx # not needed
        self.reset()

    def get_task_usages(self):
        cpu_usage_col = self.attr_idx['cpu_rate']
        mem_usg_col = self.attr_idx['can_mem_usg']
        real_cpu_usg = self.train_data[self.episode_no][self.i][cpu_usage_col]
        real_mem_usg = self.train_data[self.episode_no][self.i][mem_usg_col]
        return real_cpu_usg, real_mem_usg

    def reset(self):
        # We will be using this counter for penalty of the episode
        self.overshoot_counter = 0
        # Invokes Rule 1 and Rule2 in Step
        self.wait_action = 0
        # Total Number of Machines we train for
        self.nb_w_nodes = GYM_ENV_CFG['NB_NODES']
        # 2 (Cpu_req, mem_req)
        self.nb_dim = GYM_ENV_CFG['NB_RES_DIM']
        # used in Machine Update in State
        self.machine_status = {}
        self.sum_wait = 0
        self.reward = 0
        self.done = False
        # Indicator of Current Time in the Environment
        self.clock_time = 0
        # "Task_No":Time_for_Task captures the no of running task
        self.task_end_time = {}
        # Stores the History of Task(running on whic machine with what usages)
        self.memory = {}
        self.i = 0  # steps excluding waiting action
        self.j = 0  # steps including wait task
        # Sum of Rewards over one episode
        self.cum_reward = 0
        # wait action is supposed to be 0 no.
        self.actions = self.nb_w_nodes + 1
        # used in termination condn of episode
        self.max_steps_current_epi = 0
        # used for masking infeasible machines
        self.machine_mask = []

        self.max_end_time = 0
        self.wait_time = 0
        self.episode_duration = 0

        # updated every step records free limits of each machine
        self.machine_capacity = {}

        for machine in range(self.nb_w_nodes):
            # Initialized with Original Capacities in the beginning
            self.machine_capacity[machine] = machine_limits(machine)

        self.max_number_of_tasks_per_job = []
        for i in self.train_data.keys():
            self.max_number_of_tasks_per_job.append(len(self.train_data[i]))

        # Used for Initializing state v1
        self.max_no_task = max(self.max_number_of_tasks_per_job)
        self.current_epi_data = self.train_data[self.episode_no]
        # Assign Task request Time Normalised
        epi_durs = self.all_episodes_duration[self.episode_no]  # durations for all tasks
        # Normalised Duration of Tasks added in 1 index of state v1
        norm_duration = [epi_durs[i] / max(epi_durs) for i in range(len(epi_durs))]
        # length of current episode
        current_task_len = len(norm_duration)

        if not self.one_task: # State v1
            self.cols_state = self.nb_dim * 2 + self.nb_w_nodes * 2 + (self.nb_w_nodes) + 1  # 29
            self.nb_rows = self.max_no_task
            self.state = np.zeros((self.nb_rows, self.cols_state))  # initialization of state
            self.state[:current_task_len, 1] = norm_duration  #
            # Assiging Cpu_req, mem_req for current task
            for i, j in enumerate(self.train_data[self.episode_no]):
                self.state[i, self.nb_dim:self.nb_dim * 2] = \
                    j[self.nb_dim:self.nb_dim * 2]  # (Task placed, task_time, cpu_req, mem_req, 16usages, 8 oneHot, Done

            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(1, self.max_no_task, self.cols_state),
                                                dtype=np.float64)
        else: # State V2
            self.cols_state = self.nb_w_nodes
            # Cpureq+Mem_req+Cpu_Usages + Mem Usages + Cpu free limits + Mem Free limits
            self.nb_rows = (self.nb_dim + 1) * 2
            self.state = np.zeros((self.nb_rows, self.cols_state))

            self.moveto_next_task()
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(1, self.nb_rows, self.cols_state),
                                                dtype=np.float64)
        # 8 + 1(Wait)
        self.action_space = spaces.Discrete(self.nb_w_nodes + 1)
        for idx in range(self.nb_w_nodes):
            self.machine_status[idx] = []

        return copy.deepcopy(np.expand_dims(self.state, 0)) # Reshape state for ACME

    def moveto_next_task(self):
        # Assigning Cpu Req and Mem req in State space
        cpu_req, mem_req = self.current_epi_data[self.i][self.nb_dim:self.nb_dim * 2]
        # Row 0 -> Cpu Requests
        self.state[0] = cpu_req
        # Row1 -> Mem Requests
        self.state[1] = mem_req

    def step(self, action):

        self.j += 1  # Counter to Track all steps including wait
        action = int(action)
        cpu_usage, mem_usage = self.get_task_usages()  # usages of current task
        # Duration of current Task Execution
        time_left_for_task = self.all_episodes_duration[self.episode_no][self.i]
        info = {}
        # Free Percentage of Available Machines
        percentage_machine_used = self.calculate_percent_machine()
        # not allow it
        over_util_penalty = over_util_reward(self.overshoot_counter)

        # Rule 1: if we took wait action and there is no task running : len(task_end_time == 0)
        if (action == self.wait_action) and len(self.task_end_time) == 0:
            # We dont make any change in the environment
            self.reward = -10
            #self.cum_reward += self.reward
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Precentage"] = percentage_used_machine

        # Rule 2 : If we took wait action and tasks are running
        elif action == self.wait_action:

            # Extract Tasks with Minimum Ending Time
            min_end_time = min(self.task_end_time.values())
            # retrieve all the tasks with min end time
            tasks_with_minimum_time = [k for k, v in self.task_end_time.items() if v == min_end_time]
            # we complete the Tasks with minimum Time Value
            for i in (tasks_with_minimum_time):
                # retrieve the history of running tasks
                machine_no, cpu_usage, mem_usage = list(self.memory[i].values())
                usages = [cpu_usage, mem_usage] # Added to the Machine Free Limit
                self.change_in_machine_capacity(action=machine_no, usages=usages)

            # we change the current clock to min. of task running
            self.clock_time = min_end_time

            if not self.one_task: # State V1
                self.update_one_hot_encoding(task_index=tasks_with_minimum_time, remove=True)

            self.update_state(wait_flag=True, task=tasks_with_minimum_time)
            [self.task_end_time.pop(key) for key in tasks_with_minimum_time]  # here we pop out the keys with min values

            # Here we will give the wait_reward + over_util_penalty

            self.reward = calculate_wait_reward(len(tasks_with_minimum_time))

            #self.cum_reward += self.reward

            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine

        else:
            # Since Machine in State ranges from (0-7)
            action -= 1

            if not self.one_task:
                self.update_one_hot_encoding(action=action)

            # Capture history of Task
            self.memory[self.i] = {"Machine_No": action,
                                   "cpu_usage": cpu_usage,
                                   "mem_usage": mem_usage,
                                   }

            # Used for updating machine free limits
            self.machine_status[action].append({
                "cpu": cpu_usage,
                "mem": mem_usage,
                "rem_time": time_left_for_task,
                "task_id": self.i,
                "cpu_req": self.train_data[self.episode_no][self.i][self.attr_idx['cpu_req']],
                "mem_req": self.train_data[self.episode_no][self.i][self.attr_idx['mem_req']]
            })
            # used for updating the machine free limits
            usages = [cpu_usage, mem_usage]
            self.change_in_machine_capacity(action=action, usages=usages, placed=True)
            self.update_state()
            self.task_end_time[self.i] = time_left_for_task + self.clock_time

            if not self.one_task:
                usage = list(self.state[self.i, self.nb_dim * 2:(self.nb_dim * 2 + self.nb_w_nodes) + self.nb_w_nodes])

            else:
                usage = []
                cpu_usg = list(self.state[2])
                mem_usg = list(self.state[3])
                usage = cpu_usg + mem_usg

            if DRL_CFG['reward_type'] == 'under_util':
                self.reward = under_util_reward(usage)

            elif DRL_CFG['reward_type'] == 'simple':
                self.reward = get_intermediate_reward(action=action, usages=usage, updated_capacities=self.machine_capacity)


           # self.cum_reward += self.reward
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
            self.i +=1
            self.machine_status = {}

            for key in range(self.nb_w_nodes):
                self.machine_status[key] = []

            if self.one_task:
                self.moveto_next_task()
             # increment only when we place task

        if self.no_more_steps():  # or self.termination_conditon_waiting():
            self.done = True
            max_end_time = list(self.all_episodes_duration[self.episode_no])
            max_end_time = max(max_end_time) # here we extract Task which requires max time to run
            self.max_end_time = max_end_time
            self.episode_duration = max(self.task_end_time.values())
            self.wait_time = self.episode_duration - self.max_end_time

            #self.reward = episode_end_reward(task_end_time=self.task_end_time, clock_time=self.clock_time, \
             #                                max_end_time = max_end_time)

            self.reward = under_util_reward(usage)

            for machine in range(self.nb_w_nodes):
                cpu_limit, memory_limit = machine_limits(machine)
                self.machine_capacity[machine] = [cpu_limit, memory_limit]

            info = self.get_metric()
            self.episode_no += 1
            #self.cum_reward += self.reward

        return np.expand_dims(self.state, 0), float(self.reward), self.done, info


    def change_in_machine_capacity(self, action, usages, placed=False):

        if placed:
            capacity = self.machine_capacity[action]
            # diff = capacity - usages
            diff = [x - y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = diff
        else:
            # If we Remove Task From machine we add the usages it made while running
            capacity = self.machine_capacity[action]
            add = [x + y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = add



    def update_one_hot_encoding(self, action=None, task_index=None, remove=False):

        if remove:
            for task in task_index:
                action = self.memory[task]['Machine_No']  # we extract the machine no on which the\
                state_one_hot = self.state[task, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
                state_one_hot[action] = 0.0  # we remove the task from machine in state
                self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1] = state_one_hot
        else:

            state_one_hot = self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
            state_one_hot[action] = 1.0 # Indicator if task is running on this machine
            self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1] = state_one_hot

    def update_free_machine_limits(self):
        for machine in self.machine_capacity:
            free_cpu, free_mem = self.machine_capacity[machine]
            self.state[4,machine] = free_cpu
            self.state[5,machine] = free_mem

    def update_state(self, wait_flag=False, task=None):  # here we pass the task with min duration
        if not wait_flag:

            for machine in self.machine_status:
                for item in self.machine_status[machine]:
                    machine_cpu_usg = item['cpu']
                    machine_mem_usg = item['mem']
                    length_of_current_episode = len(self.all_episodes_duration[self.episode_no])
                    if not self.one_task:
                        self.state[:length_of_current_episode,
                        self.nb_dim * 2 + (machine)] += machine_cpu_usg  # Cpu Usage Col

                        self.state[:length_of_current_episode,
                        self.nb_dim * 2 + (machine + self.nb_w_nodes)] += machine_mem_usg  # mem usage Col

                        self.state[self.i][0] = 1  # Assigns to Task Placed as one

                    else:
                        self.state[2][machine] += machine_cpu_usg
                        self.state[3][machine] += machine_mem_usg
                        self.update_free_machine_limits()

        else:
            # when wait action Taken we remove the Task with minimum end time
            for key in task:
                machine_no, cpu_usage, mem_usage = list(self.memory[key].values())
                length_of_current_episode = len(self.all_episodes_duration[self.episode_no])
                if not self.one_task:
                    self.state[:length_of_current_episode, self.nb_dim * 2 + (machine_no)] -= cpu_usage
                    self.state[:length_of_current_episode, self.nb_dim * 2 + (machine_no) + self.nb_w_nodes] -= mem_usage
                    self.state[key][0] = 0  # Placed Removed
                    self.state[key][-1] = 1.0  # Done Incremented
                else:
                    self.state[2, machine_no] -= cpu_usage
                    self.state[3, machine_no] -= mem_usage
                    self.update_free_machine_limits()

    def get_valid_action_mask(self):
        x = np.ones(GYM_ENV_CFG['NB_NODES'] + 1) # 8
        cpu_usage, mem_usage= self.train_data[self.episode_no][self.i][4:6]
        x[0]= 0
        for element in self.machine_capacity:
            cpu_capacity = self.machine_capacity[element][0]
            mem_capacity     = self.machine_capacity[element][1]
            if (cpu_capacity < cpu_usage) or (mem_capacity < mem_usage):

                x[1+element] = 0
                x[0] = 1 # we enable waiting here
                # Counter which tells the number of times such state was observed
                self.overshoot_counter +=1

        return x

    def get_metric(self):
        info = {}
        # Log the reward
        if self.no_more_steps():  # or self.termination_conditon_waiting():
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
            #info["Cumulative_Reward"] = self.cum_reward
        else:
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
            info["Step-Reward"] = self.reward
        return info

    '''def termination_conditon_waiting(self):
        maximum_waiting_current_epi = len(self.train_data[self.episode_no]) * 2
        return self.j >= maximum_waiting_current_epi'''

    def calculate_task_completed_epi(self):
        total_task_current_epi = len(self.train_data[self.episode_no]) - 1
        percent_of_task_completed = (self.i / total_task_current_epi) * 100
        total_steps_including_waiting = self.j
        total_steps_excluding_wait = self.i
        output_info = [percent_of_task_completed, total_task_current_epi, total_steps_including_waiting,
                       total_steps_excluding_wait]
        return output_info

    def calculate_percent_machine(self):
        percentage_per_machine = {}
        for machine in range(self.nb_w_nodes):  # 8
            cpu_limit, memory_limit = machine_limits(machine)  # orignal limit of Machines
            changed_cpu_limit, changed_mem_limit = self.machine_capacity[machine]  # Changed Limit of the Machine
            percentage_cpu = ((cpu_limit - changed_cpu_limit) / cpu_limit) * 100  # percentage of Machine Used after placement5
            percetage_mem = ((memory_limit - changed_mem_limit) / memory_limit) * 100
            percentage_per_machine[machine] = [percentage_cpu, percetage_mem]
        return percentage_per_machine

    # this function set to True would randomly initialize 30 % machines with min value
    def random_initialize_machine(self, random_initialize):
        if random_initialize:
            self.machine_mask = np.random.choice([True, False], size=self.nb_w_nodes, p=[0.6, 0.4])
            machine_list = list(np.where(self.machine_mask == True)[0])
            for machine in machine_list:
                random_cpu = random.uniform(0, 0.3)
                random_mem = random.uniform(0, 0.2)
                length_of_current_task = len(self.all_episodes_duration[self.episode_no])
                self.state[:length_of_current_task, self.nb_dim * 2 + machine] = random_cpu
                self.state[:length_of_current_task, self.nb_dim * 2 + self.nb_w_nodes + machine] = random_mem

    # First Termination condition
    def no_more_steps(self):

        self.max_steps_current_epi = len(self.train_data[self.episode_no]) - 1
        return self.i == self.max_steps_current_epi

    def gen_plot(self, timestep=None, path_to_dir=None):
        state = self.state
        timestep = self.i
        percentage_used_machine = self.calculate_percent_machine()
        cpu_usages = []
        mem_usages = []
        for key in percentage_used_machine:
            cpu_usages.append(percentage_used_machine[key][0])
            mem_usages.append(percentage_used_machine[key][1])

        # cpu_usgages = state[0][4:4 + 8]
        # mem_usages = state[0][4 + 8:4 + 8 * 2]

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
        plt.xticks(r + width / 2, ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])
        plt.ylim(0, 100)
        plt.legend()
        # plt.show()
        name = str(timestep) + ".jpeg"
        fig.savefig(os.path.join(path_to_dir, name), bbox_inches='tight', dpi=150)
        plt.close()

    def make_gif(self, path):
        png_dir = path
        images = []
        files = os.listdir(png_dir)
        sorted_list = natsort.natsorted(files, reverse=False)

        for file_name in sorted_list:
            if file_name.endswith('.jpeg'):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))

        gif_name = "movie.gif"
        imageio.mimsave(os.path.join(path, gif_name), images, format='GIF', duration=0.4)

# The reward would be taken from the task with last step in the task end time
# Include the one Hot encoding for the task
# Include the Task Time for the task
import copy
import random
import gym
import numpy as np
from config import *
from gym import spaces
np.random.seed(GYM_ENV_CFG['SEED'])
class customEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 train_data,
                 task_duration,
                 state_idx,
                 attr_idx,
                 start_idx=0):
        super(customEnv, self).__init__()
        self.train_data = train_data
        self.all_episodes_duration = task_duration
        self.state_idx = state_idx
        self.attr_idx = attr_idx
        self.start_idx = start_idx
        self.wait_action = 0
        self.episode_no = 0
        self.reward = 0
        self.done = False
        self.clock_time = 0
        self.task_end_time = {}
        self.memory = {}
        self.i = 0
        self.machine_status = {}
        self.nb_w_nodes = GYM_ENV_CFG['NB_NODES']
        self.nb_dim = GYM_ENV_CFG['NB_RES_DIM']
        # The cols we need to include in state: TaskDone, TaskReq_time, Request,
        # Usages+Onehot+done : 2+2+2+
        self.cols_state = self.nb_dim * 2 + self.nb_w_nodes * 2 + self.nb_w_nodes + 1
        self.state = np.zeros(0)
        self.actions = self.nb_w_nodes + 1  # 1 adding wait in actions
        self.max_number_of_tasks_per_job = []
        self.max_steps_current_epi = 0
        self.machine_mask = []
        for i in self.train_data.keys():
            self.max_number_of_tasks_per_job.append(len(self.train_data[i]))
        self.max_no_task = max(self.max_number_of_tasks_per_job)
        self.action_space = spaces.Discrete(self.nb_w_nodes + 1)
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(1, self.max_no_task, self.cols_state),
                                             dtype = np.float64)

        self.reset()

    def get_task_usages(self):
        cpu_usage_col = self.attr_idx['cpu_rate']
        mem_usg_col = self.attr_idx['can_mem_usg']
        real_cpu_usg = self.train_data[self.episode_no][self.i][cpu_usage_col]
        real_mem_usg = self.train_data[self.episode_no][self.i][mem_usg_col]
        return real_cpu_usg, real_mem_usg

    def update_machine_state_rem_time(self):
        for key in self.machine_status:
            running_task = self.machine_status[key]
            run_task_temp = []
            for t_dict in running_task:
                t_dict['rem_time'] -= 1
                if t_dict['rem_time'] > 0:
                    run_task_temp.append(t_dict)
            self.machine_status[key] = run_task_temp

    def update_state(self, wait_flag=False, task=None):
        if task != None \
                and  task > len(self.all_episodes_duration[self.episode_no]):
            return

        if not wait_flag:
            for machine in self.machine_status:
                for item in self.machine_status[machine]:
                    machine_cpu_usg = item['cpu']
                    machine_mem_usg = item['mem']
                    length_of_current_task = len(self.all_episodes_duration[self.episode_no])
                    self.state[:length_of_current_task, self.nb_dim * 2 + (machine - 1)] += machine_cpu_usg  # Cpu Usage Col
                    self.state[:length_of_current_task, self.nb_dim * 2 + (machine + self.nb_w_nodes - 1)] += machine_mem_usg  # mem usage Col
            self.state[self.i][0] = 1  # Assigns to Task Placed as one
        else:
            machine_no, cpu_usage, mem_usage = list(self.memory[task].values())
            length_of_current_task = len(self.all_episodes_duration[self.episode_no])
            self.state[:length_of_current_task, self.nb_dim * 2 + (machine_no - 1)] -= cpu_usage
            self.state[:length_of_current_task, self.nb_dim * 2 + (machine_no - 1) + self.nb_w_nodes] -= mem_usage
            self.state[task][0] = 0  # Placed Removed
            self.state[task][-1] = 1.0  # Done Incremented

    def step(self, action):
        action = int(action)
        cpu_usage, mem_usage = self.get_task_usages()
        time_left_for_task = self.all_episodes_duration[self.episode_no][self.i]
        # Rule1
        if (action == self.wait_action) and len(self.task_end_time) == 0:
            state = copy.deepcopy(self.state)
            self.state = state
            self.reward = 0

        elif action == self.wait_action:
            task_with_min_time = min(self.task_end_time, key=self.task_end_time.get)
            min_end_time = self.task_end_time[task_with_min_time]
            self.clock_time = min_end_time
            self.update_one_hot_encoding(action=action, task_index=task_with_min_time, remove=True)
            self.update_state(wait_flag=True, task=task_with_min_time)
            self.task_end_time.pop(task_with_min_time)
            self.reward = 0
        else:
            self.update_one_hot_encoding(action)
            self.memory[self.i] = {"Machine_No": action,
                                   "cpu_usage": cpu_usage,
                                   "mem_usage": mem_usage,
                                   }
            self.machine_status[action].append({
                "cpu": cpu_usage,
                "mem": mem_usage,
                "rem_time": time_left_for_task,
                "task_id": self.i,
                "cpu_req": self.train_data[self.episode_no][self.i][self.attr_idx['cpu_req']],
                "mem_req": self.train_data[self.episode_no][self.i][self.attr_idx['mem_req']]
            })
            self.update_state()
            self.update_machine_state_rem_time()
            self.task_end_time[self.i] = time_left_for_task + self.clock_time
            usage = list(self.state[self.i, self.nb_dim*2:(self.nb_dim*2+self.nb_w_nodes)+self.nb_w_nodes])
            self.reward = self.get_intermediate_reward(action=action, usages=usage)
            self.i += 1  # increment only when we place task

        if self.no_more_steps():
            self.done = True
            self.reward = self.episode_end_reward()
            self.episode_no += 1

        return copy.deepcopy(np.expand_dims(self.state,0)), float(self.reward), self.done, {}

    def random_initialize_machine(self):
        self.machine_mask = np.random.choice([True, False], size=self.nb_w_nodes, p=[0.6, 0.4])
        machine_list = list(np.where(self.machine_mask == True)[0])
        for machine in machine_list:
            random_cpu = random.uniform(0, 0.3)
            random_mem = random.uniform(0, 0.2)
            length_of_current_task = len(self.all_episodes_duration[self.episode_no])
            self.state[:length_of_current_task, self.nb_dim * 2 + machine] = random_cpu
            self.state[:length_of_current_task , self.nb_dim * 2 + self.nb_w_nodes + machine] = random_mem

    def update_one_hot_encoding(self, action, task_index=None, remove=False):
        action = action - 1
        if remove:
            state_one_hot = self.state[task_index, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
            state_one_hot[action] = 0.0
        else:
            state_one_hot = self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
            state_one_hot[action] = 1.0
        self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1] = state_one_hot

    def reset(self):
        self.task_end_time = {}
        self.i = 0
        self.done = False
        for i in self.train_data.keys():
            self.max_number_of_tasks_per_job.append(len(self.train_data[i]))
        self.max_no_task = max(self.max_number_of_tasks_per_job)
        self.state = np.zeros((self.max_no_task, self.cols_state))
        # Initialize CPU req and Mem Req
        for i, j in enumerate(self.train_data[self.episode_no]):
            self.state[i, self.nb_dim:self.nb_dim * 2] = \
                j[self.nb_dim:self.nb_dim * 2]  # (Task placed, task_time, cpu_req, mem_req, 16usages, 8 oneHot, Done
        # Since action 0 is wait Increment Machines with 1
        for idx in range(self.nb_w_nodes):
            self.machine_status[idx + 1] = []
        # Assign Task request Time Normalised
        epi_durs = self.all_episodes_duration[self.episode_no]
        norm_duration = [epi_durs[i] / max(epi_durs) for i in range(len(epi_durs))]
        current_task_len = len(norm_duration)
        self.state[:current_task_len, 1] = norm_duration

        self.random_initialize_machine()
        return copy.deepcopy(np.expand_dims(self.state,0))

    def get_intermediate_reward(self, action, usages):
        usage_2d = [usages[i] + usages[i + self.nb_w_nodes] for i in range(self.nb_w_nodes)]
        usage_2d = np.insert(usage_2d, 0, 0.0)  # we insert the wait action here
        least_used_machines = list(np.where(usage_2d == min(usage_2d))[0])
        machine_cpu_cap, machine_mem_cap = self.machine_limits(action)
        total_cap = machine_cpu_cap + machine_mem_cap
        reward = 0
        if action in least_used_machines:
            reward = -10
        elif usage_2d[action] > total_cap:
            reward = -5
        else:
            reward = 1
        return reward

    def episode_end_reward(self):
        reward = 100 * (self.clock_time / max(self.task_end_time.values()))
        return reward

    # First Termination condition
    def no_more_steps(self):
        self.max_steps_current_epi = len(self.train_data[self.episode_no]) - 1
        return self.i >= self.max_steps_current_epi

    def machine_limits(self, machine):
        machine_cpu_type = GYM_ENV_CFG['MC_CAP'][machine]
        machine_cpu_cap = GLOBAL_CFG['MC_CAP_VALUES'][machine_cpu_type]
        machine_mem_type = GYM_ENV_CFG['MM_CAP'][machine]
        machine_mem_cap = GLOBAL_CFG['MM_CAP_VALUES'][machine_mem_type]
        return machine_cpu_cap, machine_mem_cap

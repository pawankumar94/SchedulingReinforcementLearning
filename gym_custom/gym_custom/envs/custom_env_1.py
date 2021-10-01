# Todo Include the Penalty Term for teh Reward Calculation
# Todo Include wait Action also in the output of the Neural Network
import copy
import random
import gym
import numpy as np
from config import *
from gym import spaces
import matplotlib.pyplot as plt
np.random.seed(GYM_ENV_CFG['SEED'])
class customEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 train_data,
                 task_duration,
                 state_idx,
                 attr_idx,
                 start_idx=0,
                 random_initialize = False):
        super(customEnv, self).__init__()
        self.environment_time = 0
        self.train_data = train_data  # loading the dataset
        self.all_episodes_duration = task_duration
        self.state_idx = state_idx
        self.attr_idx = attr_idx  # cpu:0 and memory:1
        self.start_idx = start_idx
        self.wait_action = 0
        self.episode_no = 0
        self.reward = 0
        self.done = False
        self.clock_time = 0
        self.task_end_time = {}  # "Task_No":Time_for_Task captures the no of running task
        self.memory = {}
        self.i = 0  # steps excluding waiting action
        self.j = 0  # steps including wait task
        self.random_initialize = random_initialize  # Flag to determine if some machines should\
        # be initialized with some value in beginning
        self.machine_status = {}
        self.nb_w_nodes = GYM_ENV_CFG['NB_NODES']  # 8
        self.nb_dim = GYM_ENV_CFG['NB_RES_DIM']  # 2
        # The cols we need to include in state: TaskDone, TaskReq_time, Request,
        # Usages+Onehot+done : 2+2+2+
        self.cols_state = self.nb_dim * 2 + self.nb_w_nodes * 2 + (self.nb_w_nodes) + 1  # 29
        self.state = np.zeros(0)
        self.actions = self.nb_w_nodes + 1   # wait action is supposed to be 0 no.
        self.max_number_of_tasks_per_job = []
        self.max_steps_current_epi = 0
        self.machine_mask = []

        for i in self.train_data.keys():
            self.max_number_of_tasks_per_job.append(len(self.train_data[i]))
        self.machine_capacity = {}
        self.max_no_task = max(self.max_number_of_tasks_per_job)
        self.action_space = spaces.Discrete(self.nb_w_nodes + 1)  #
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

    # Not Important Function
    '''def update_machine_state_rem_time(self):
        for key in self.machine_status:
            running_task = self.machine_status[key]
            run_task_temp = []
            for t_dict in running_task:
                t_dict['rem_time'] -= 1
                if t_dict['rem_time'] > 0:
                    run_task_temp.append(t_dict)
            self.machine_status[key] = run_task_temp
'''
    def update_state(self, wait_flag=False, task=None):

        '''if task != None \
                and  task > len(self.all_episodes_duration[self.episode_no]):
            return'''

        if not wait_flag:
            for machine in self.machine_status:
                for item in self.machine_status[machine]:
                    machine_cpu_usg = item['cpu']
                    machine_mem_usg = item['mem']
                    length_of_current_episode = len(self.all_episodes_duration[self.episode_no])
                    self.state[:length_of_current_episode, self.nb_dim * 2 + (machine)] += machine_cpu_usg  # Cpu Usage Col
                    self.state[:length_of_current_episode, self.nb_dim * 2 + (machine + self.nb_w_nodes)] += machine_mem_usg  # mem usage Col
            self.state[self.i][0] = 1  # Assigns to Task Placed as one

        else:

            for key in task:
                machine_no, cpu_usage, mem_usage = list(self.memory[key].values())
                length_of_current_episode = len(self.all_episodes_duration[self.episode_no])
                self.state[:length_of_current_episode, self.nb_dim * 2 + (machine_no)] -= cpu_usage
                self.state[:length_of_current_episode, self.nb_dim * 2 + (machine_no) + self.nb_w_nodes] -= mem_usage
                self.state[key][0] = 0  # Placed Removed
                self.state[key][-1] = 1.0  # Done Incremented

    def step(self, action):
        #action_capacity = self.machine_capacity[action]
        self.j+= 1  # total number of steps taken inluding waiting
        action = int(action)
        cpu_usage, mem_usage = self.get_task_usages()  # usages of current task
        time_left_for_task = self.all_episodes_duration[self.episode_no][self.i]  # duration of \
        # current task
        info = {}

        # Rule 1: if we took wait action and there is no task running : len(task_end_time == 0)
        if (action == self.wait_action) and len(self.task_end_time) == 0:
            state = copy.deepcopy(self.state)
            self.state = state
            self.reward = 0
            #percentage_used_machine = self.calculate_percent_machine()
            #info["machine-Used-Percentage"] = percentage_used_machine

        elif action == self.wait_action:
            min_end_time = min(self.task_end_time.values())
            tasks_with_minimum_time = [k for k, v in self.task_end_time.items() if v==min_end_time]
            #task_with_min_time = min(self.task_end_time, key=self.task_end_time.get)
            #min_end_time = self.task_end_time[task_with_min_time]

            for i in (tasks_with_minimum_time):
                machine_no, cpu_usage, mem_usage = list(self.memory[i].values())
                usages = [cpu_usage, mem_usage]
                self.change_in_machine_capacity(action=machine_no, usages=usages)

            self.clock_time = min_end_time  # we change the current clock to min. of task running
            self.update_one_hot_encoding(task_index=tasks_with_minimum_time, remove=True)
            self.update_state(wait_flag=True, task=tasks_with_minimum_time)
            #self.task_end_time.pop(task_with_min_time)
            [self.task_end_time.pop(key) for key in tasks_with_minimum_time]  # here we pop out the keys with min values
            self.reward = 0.5
          #  percentage_used_machine = self.calculate_percent_machine()
          #  info["machine-Used-Percentage"] = percentage_used_machine

        else:
            action -= 1
            self.update_one_hot_encoding(action = action)
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
   #           self.update_machine_state_rem_time()
            usages = [cpu_usage, mem_usage]
            self.change_in_machine_capacity(action = action, usages=usages, placed= True)
            self.task_end_time[self.i] = time_left_for_task + self.clock_time
            usage = list(self.state[self.i, self.nb_dim*2:(self.nb_dim*2+self.nb_w_nodes)+self.nb_w_nodes])
            #percent_used_machines = self.calculate_percent_machine(usage)
            self.reward = self.get_intermediate_reward(action=action, usages=usage)
            self.i += 1  # increment only when we place task
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
           # self.gen_plot()

        if self.no_more_steps() or self.termination_conditon_waiting():
            self.done = True
            self.reward = self.episode_end_reward()
            ''''percentage_used_machine = self.calculate_percent_machine()
            info["Final_Machines_Percentage_usage"] = percentage_used_machine
            percent_of_task_completed, total_no_of_tasks\
                , total_steps_including_waiting, total_steps_excluding_wait \
                = self.calculate_task_completed_epi()
            info["Percentage_Task_Completed"] = percent_of_task_completed
            info["Total_Task_Episode"] = total_no_of_tasks
            info["Steps_Including_Wait"] = total_steps_including_waiting
            info["Steps_Without_Wait"] = total_steps_excluding_wait
            info["Wait_steps_taken"] = total_steps_including_waiting - total_steps_excluding_wai'''
            self.episode_no += 1
            #info["Percentage_Task_Completed"] = self.calculate_task_completed_epi()
        info = self.get_metric()
        return copy.deepcopy(np.expand_dims(self.state,0)), float(self.reward), self.done, info

    def get_metric(self):
        info = {}
        # Log the reward
        if self.no_more_steps() or self.termination_conditon_waiting():
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
        else:
            percentage_used_machine = self.calculate_percent_machine()
            info["machine-Used-Percentage"] = percentage_used_machine
            info["Step-Reward"] = self.reward
        return info


    def termination_conditon_waiting(self):
        maximum_waiting_current_epi = len(self.train_data[self.episode_no]) * 2
        return self.j >= maximum_waiting_current_epi

    def calculate_task_completed_epi(self):
        total_task_current_epi = len(self.train_data[self.episode_no])-1
        percent_of_task_completed = (self.i / total_task_current_epi)*100
        total_steps_including_waiting = self.j
        total_steps_excluding_wait = self.i
        output_info = [percent_of_task_completed, total_task_current_epi, total_steps_including_waiting, total_steps_excluding_wait]
        return output_info

    def calculate_percent_machine(self):
        percentage_per_machine = {}
        for machine in range(self.nb_w_nodes):
            cpu_limit, memory_limit = self.machine_limits(machine)
            changed_cpu_limit, changed_mem_limit = self.machine_capacity[machine]
            percentage_cpu = ((cpu_limit - changed_cpu_limit) / cpu_limit) * 100
            percetage_mem = ((memory_limit- changed_mem_limit) / memory_limit) * 100
            percentage_per_machine[machine] = [percentage_cpu, percetage_mem]
        return percentage_per_machine

    def random_initialize_machine(self,random_initialize):
        if random_initialize:
            self.machine_mask = np.random.choice([True, False], size=self.nb_w_nodes, p=[0.6, 0.4])
            machine_list = list(np.where(self.machine_mask == True)[0])
            for machine in machine_list:
                random_cpu = random.uniform(0, 0.3)
                random_mem = random.uniform(0, 0.2)
                length_of_current_task = len(self.all_episodes_duration[self.episode_no])
                self.state[:length_of_current_task, self.nb_dim * 2 + machine] = random_cpu
                self.state[:length_of_current_task , self.nb_dim * 2 + self.nb_w_nodes + machine] = random_mem

    def change_in_machine_capacity(self, action, usages, placed=False):
        if placed:
            capacity = self.machine_capacity[action]
           # diff = capacity - usages
            diff = [x - y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = diff
        else:
            capacity = self.machine_capacity[action]
            add = [x + y for x, y in zip(capacity, usages)]
            self.machine_capacity[action] = add
        return

    def update_one_hot_encoding(self, action= None, task_index=None, remove=False):
        if remove:
            for task in task_index:
               action = self.memory[task]['Machine_No']  # we extract the machine no on which the\
               state_one_hot = self.state[task, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
               state_one_hot[action] = 0.0  # we remove the task from machine in state
               self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1] = state_one_hot

            '''for key in task_index:
                state_one_hot = self.state[key, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
                state_one_hot[action] = 0.0'''

        else:
            state_one_hot = self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1]
            state_one_hot[action] = 1.0
            self.state[self.i, self.nb_dim * 2 + self.nb_w_nodes * 2:-1] = state_one_hot

    def reset(self):
        self.task_end_time = {}
        self.i = 0  # steps taken considering tasks
        self.j = 0  # steps including wait action
        self.done = False

        for i in self.train_data.keys():
            self.max_number_of_tasks_per_job.append(len(self.train_data[i]))  # this list \
            # contains the length of each episode

        for machine in range(self.nb_w_nodes):
            self.machine_capacity[machine] = self.machine_limits(machine)

        self.max_no_task = max(self.max_number_of_tasks_per_job)  # maximum overall episodes
        self.state = np.zeros((self.max_no_task, self.cols_state))  # initialization of state

        # Initialize CPU req and Mem Req
        for i, j in enumerate(self.train_data[self.episode_no]):
            self.state[i, self.nb_dim:self.nb_dim * 2] = \
                j[self.nb_dim:self.nb_dim * 2]  # (Task placed, task_time, cpu_req, mem_req, 16usages, 8 oneHot, Done

        for idx in range(self.nb_w_nodes):
            self.machine_status[idx] = []  # used in machine_update()

        # Assign Task request Time Normalised
        epi_durs = self.all_episodes_duration[self.episode_no]  # durations for all tasks
        norm_duration = [epi_durs[i] / max(epi_durs) for i in range(len(epi_durs))]  # normalised \
        current_task_len = len(norm_duration)  # length of current episode
        self.state[:current_task_len, 1] = norm_duration
        self.random_initialize_machine(random_initialize=self.random_initialize)

        return copy.deepcopy(np.expand_dims(self.state,0))

    def get_intermediate_reward(self, action, usages):
        usage_2d = [usages[i] + usages[i + self.nb_w_nodes] for i in range(self.nb_w_nodes)]
     #   usage_2d = np.insert(usage_2d, 0, 0.0)  # we insert the wait action here
        least_used_machines = list(np.where(usage_2d == min(usage_2d))[0])
       # machine_cpu_cap, machine_mem_cap = self.machine_limits(action)
        updated_cpu_cap, updated_memory_cap = self.machine_capacity[action]
        #total_cap = machine_cpu_cap + machine_mem_cap
        reward = 0
        if action in least_used_machines:
            reward = -10
        #elif usage_2d[action] > total_cap:
        #    reward = -5
        elif (updated_cpu_cap<=0) or (updated_memory_cap<=0):
            reward = -20
        else:
            reward = 1
        return reward

    def episode_end_reward(self):
        if not all(self.task_end_time.values()):
            reward = 100 * (self.clock_time / max(self.task_end_time.values()))
        else:
            #reward = 100 * (self.clock_time / max(self.task_end_time.values()))
            reward = 10
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
        return [machine_cpu_cap, machine_mem_cap]

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
        #mem_usages = state[0][4 + 8:4 + 8 * 2]
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
        plt.show()
   #     name = str(timestep) + ".jpeg"
   #     print(name)
   #     fig.savefig(os.path.join(path_to_dir, name), bbox_inches='tight', dpi=150)

    def make_gif(self,path):
        png_dir = path
        images = []
        for file_name in sorted(os.listdir(png_dir)):
            if file_name.endswith('.jpeg'):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))
        gif_name = "movie.gif"
        imageio.mimsave(os.path.join(path, gif_name), images)
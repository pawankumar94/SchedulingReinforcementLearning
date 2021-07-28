import numpy as np
import os
import pickle

class Env:
    def __init__(self):
        self.machine_state = []
        self.state = []
        self.timestep = []
        self.sample_list = []
        self.next_state = []
        self.next_dict = {}
        self.action = 0
        self.task_number = 0
        self.cpu = 0
        self.cpu_usage = 0
        self.memory_usage = 0
        self.done = 0
        self.memory = {}

    def read_pickle(self, name, path, count):
        infile = open(path + '/' + name, 'rb')
        pickle_dict = pickle.load(infile)
        try:
            next_name = self.sample_list[count]
            infile = open(path + '/' + next_name, 'rb')
            self.next_dict = pickle.load(infile)
        except:
            pass
        return pickle_dict

    @staticmethod
    def reset(counter, arr):
        return arr[counter]

    def list_of_pickles(self, path):
        self.sample_list = os.listdir(path)
        return self.sample_list

    def get_state(self, task_dict):
        keys = list(task_dict.keys())
        self.action = task_dict['Action_Value']
        self.task_number = task_dict['Task_Number']
        self.state = task_dict[keys[0]][self.task_number]
        self.state = np.asarray(self.state)
        indexes = [0]
        self.state = [i for j, i in enumerate(self.state) if j not in indexes]
        self.timestep = keys[0]
        self.cpu_usage = self.state[2]
        self.memory_usage = self.state[3]
        self.done = self.state[4]
        if self.done == 0.0:
            self.done = False
        else:
            self.done = True
        if (self.cpu_usage != 0) & (self.memory_usage != 0):
            usages = [self.memory_usage, self.cpu_usage]
            self.memory[self.task_number] = usages
        return self.task_number, self.state, self.timestep, self.action, self.done

    def get_machine_state(self):
        path = "Machine_State_microsecond"
        name_machine = self.timestep + ".pickle"
        infile = open(path + '/' + name_machine, 'rb')
        pickle_dict = pickle.load(infile)
        key = list(pickle_dict.keys())
        self.machine_state = np.asarray(pickle_dict[key[0]])

    def update_machine_state(self, action):
        self.get_machine_state()
        sample_action = self.machine_state[action][0]
        row = np.where(self.machine_state[:, 0] == sample_action)
        _x = self.machine_state[row][0]
        _cpu = _x[4]
        _mem = _x[5]
        if action == 0:
            x = self.memory[self.task_number]
            _cpu = _cpu + x[0]
            _mem = _mem + x[1]
            _x[4] = _cpu
            _x[5] = _mem
            self.machine_state[row] = _x
        else:
            _cpu = abs(_cpu - self.cpu_usage)
            _mem = abs(_mem - self.memory_usage)
            _x[4] = _cpu
            _x[5] = _mem
            self.machine_state[row] = _x
        return self.machine_state

    @staticmethod
    def get_reward():
        return 1

    def step(self, action):
        task_number, self.next_state, timestep, action_1, done = self.get_state(self.next_dict)
        machine_state = self.update_machine_state(action)
        reward = self.get_reward()
        return self.next_state, machine_state, reward

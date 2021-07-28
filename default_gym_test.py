"""
Todo: Remove the Done Parameter from the Task state
Todo: Return the state and next state of shape 37 x4 array
Todo: Add one to the axis 1 in Task State when task is placed on Machine and 0 when removed
Todo: Include the Machine State in the Next State
Todo:
Todo: Discuss about the integration with DQFD and R2D3
Todo: Discuss the reward calculation in Step()
"""

"""
1. Discuss about the Issue with Episodes with just 2 timesteps( Solved)
2. Propose the idea for Data Generation Online (Solved)
3. Discuss the integration of machine state in log ( Solved)
"""

import os
from tkinter import Tcl
import time
from default_gym import Env

# Training loop
_env = Env()
path = r"/Episode_Data"
arr = os.listdir('./' + path)
arr = Tcl().call('lsort', '-dict', arr)
arr = list(arr)
counter = 0
for i in range(len(arr)):
    print("Episode", counter)  # save the episode as Unique ID
    name_of_episode = _env.reset(counter, arr)
    path_1 = './' + path + "/" + name_of_episode
    sample_list = _env.list_of_pickles(path_1)  # Number of Pickles in each Episode
    count = 0
    # Todo: Need to pickle each episode data as a log file
    for idx in sample_list:
        count += 1
        sample_dict = _env.read_pickle(idx, path_1, count)
        task_number, state, timestep, action, done = _env.get_state(sample_dict)
        print("Task_Number", task_number, "timestep:", timestep)
        print("State:", state, "Action:", action)
        ## Todo: Need to design a reward functionality
        next_state, machine_state, reward = _env.step(action)
        print("Next State:", next_state)
        ## Todo: Need to fix the Done status
        print("Episode Done:", done, "\n")
        time.sleep(1)
    print("********** End of Episode **********")
    counter += 1
    time.sleep(2)

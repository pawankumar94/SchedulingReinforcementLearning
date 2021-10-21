import numpy as np
from config import *
from data_preprocess import *

def under_util_reward(usages):
    k_u = GLOBAL_CFG['K_u']
    scale_factor = 5
    usage_2d = [usages[i] + usages[i + GYM_ENV_CFG['NB_NODES']] for i in range(GYM_ENV_CFG['NB_NODES'])]
    reward = -(np.sum(np.power(usage_2d, k_u))/scale_factor)
    return reward

def calculate_wait_reward(no_of_task_ends):
    return 0.5 * no_of_task_ends


def get_intermediate_reward(action, usages, updated_capacities):
    # take in consideration req made by task
    usage_2d = [usages[i] + usages[i + GYM_ENV_CFG['NB_NODES']] for i in range(GYM_ENV_CFG['NB_NODES'])]
    least_used_machines = list(np.where(usage_2d == min(usage_2d))[0])
    updated_cpu_cap, updated_memory_cap = updated_capacities[action]
    limit_cpu, limit_mem = machine_limits(action)
    percentage_cpu = ((limit_cpu - updated_cpu_cap) / limit_cpu) * 100  # percentage of Machine Used after placement5
    percentage_mem = ((limit_mem - updated_memory_cap) / limit_cpu) * 100
    # remove
    '''if action in least_used_machines:
        reward = -15'''

    if (percentage_mem >= 95.0) and (percentage_cpu >= 95.0):
        reward = -10

    elif (percentage_cpu >= 30.0 and  percentage_cpu <95.0) \
            and (percentage_mem >= 30.0 and  percentage_mem < 95.0) :
        reward = 10
    else:
        reward = -15

    return reward


def over_util_reward():
    return -20

def episode_end_reward(task_end_time, clock_time):
    # check for this Episode End Reward
    # if not all(self.task_end_time.values()):
    # if the dict is not empty
    # if all tasks were placed
    if task_end_time.keys():
        reward = 100 * (clock_time / max(task_end_time.values())) # replace the task end time with max (theoretical time)
    else:
        # if you skip any task
        reward = 10  # Configure this Value

    return reward

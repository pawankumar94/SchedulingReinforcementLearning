import numpy as np
import pandas as pd
import numpy as np
from config import *
import matplotlib.pyplot as plt
np.random.seed(GYM_ENV_CFG['SEED'])
def data_gen(sample_size, res_dist_dict, ratio=1):
    # give the probability of selecting [low, medium, high]
    df_cols = {}
    for resource in ['cpu_req', 'mem_req']:
        element_list = []
        if res_dist_dict[resource] == "uniform":
            probs = [1 / 3, 1 / 3, 1 / 3]
        elif res_dist_dict[resource] == "intensive":
            probs = [0.2, 0.2, 0.6]
        elif res_dist_dict[resource] == "medium":
            probs = [0.2, 0.6, 0.2]
        elif res_dist_dict[resource] == "light":
            probs = [0.6, 0.2, 0.2]
        lows = np.random.uniform(low=0, high=0.299, size=sample_size)
        lows = np.round(lows, 4)
        meds = np.random.uniform(low=0.3, high=0.599, size=sample_size)
        meds = np.round(meds, 4)
        highs = np.random.uniform(low=0.6, high=1, size=sample_size)
        highs = np.round(highs, 4)
        for i in range(sample_size):
            sample_list = np.random.choice([lows[i], meds[i], highs[i]], size=1, p=probs)
            element_list.append(sample_list[0])
        df_cols[resource] = element_list
    df_sample = pd.DataFrame(df_cols)
    # Ratio should be between Range 0-1 specifies the Percentage of utilization Made as per request
    df_sample['cpu_rate'] = df_sample['cpu_req'] * ratio
    df_sample['can_mem_usg'] = df_sample['mem_req'] * ratio
    data = np.zeros(df_sample.shape[0])
    df_sample.insert(0, 'Placed', data)  # Placement Column
    df_sample.insert(1, "Task_requested_time", data)
    df_sample.insert(len(df_sample.columns), "Done", data)  # Task completed Column
    return df_sample


# Need to Pass the single col_list(e.g df['cpu_req']) as input and col name (e.g. 'cpu_req')
def gen_visualization(df_list, col_name):
    x = pd.DataFrame(df_list)
    col = col_name
    conditions = [x[col] > 2 / 3, (x[col] >= 1 / 3) & (x[col] <= 2 / 3), x[col] < 1 / 3]
    choices = ["high", 'medium', 'low']
    x["load_class"] = np.select(conditions, choices)

def preprocess_data(df, len_data):
    task_data = df
    attr_idx, state_indices = gen_states(dataset=task_data, features_to_include=['cpu_req', 'mem_req'])
    data_subset = task_data.to_numpy()[:len_data, :]  # Length of Training Data
    return data_subset, attr_idx, state_indices

def generate_duration(data_subset, key):
    task_durs = GLOBAL_CFG['TASK_DURS_MEDIUM']
    subset_task_durs = np.random.choice(task_durs, len(data_subset[key]))
    return subset_task_durs

def gen_states(dataset, features_to_include):
    attr_idx = {}
    for idx, val in enumerate(dataset.columns.to_list()):
        attr_idx[val] = idx
    state_indices = []
    for elm in features_to_include:
        state_indices.append(attr_idx[elm])
    return attr_idx, state_indices
import pandas as pd
import numpy as np
from config import *
import matplotlib.pyplot as plt
import seaborn as sns

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

def preprocess_data(df, len_data):
    task_data = df
    attr_idx, state_indices = gen_states(dataset=task_data, features_to_include=['cpu_req', 'mem_req'])
    data_subset = task_data.to_numpy()[:len_data, :]  # Length of Training Data
    return data_subset, attr_idx, state_indices

def generate_duration(data_subset, key, length_duration):
    task_durs = length_duration
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

def generate_visualizations(synthetic_df, sample_dict):
    plt.subplots(figsize=(20, 20))
    count = 1
    for idx, type in enumerate(GLOBAL_CFG['features_to_include']):
        plt.subplot(3, 2, count)
        plt.title(sample_dict[type])
        sample_list = synthetic_df[type]
        sample_df = pd.DataFrame(sample_list)
        conditions = [sample_df[type] > 0.599, (sample_df[type] >= 0.299) \
                      & (sample_df[type] <= 0.599), sample_df[type] < 0.299]
        choices = ["high", "medium", "low"]
        sample_df["load_class"] = np.select(conditions, choices)
        sns.set(rc={'figure.figsize': (15, 5.27)})
        sns.histplot(data=sample_df, x=type, alpha=1, bins=50, hue='load_class')
        count += 1
    plt.show()

def gen_plot(state, timestep=None, path_to_dir=None):
    cpu_usgages = state[0][0][4:4 + 8]
    mem_usages = state[0][0][4 + 8:4 + 8 * 2]
    fig = plt.figure(figsize=(10, 5))
    n = GYM_ENV_CFG['NB_NODES']
    r = np.arange(n)
    width = 0.25
    plt.bar(r, cpu_usgages, color='g',
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
    plt.legend()
    plt.show()

def machine_limits(machine):
    machine_cpu_type = GYM_ENV_CFG['MC_CAP'][machine]
    machine_cpu_cap = GLOBAL_CFG['MC_CAP_VALUES'][machine_cpu_type]
    machine_mem_type = GYM_ENV_CFG['MM_CAP'][machine]
    machine_mem_cap = GLOBAL_CFG['MM_CAP_VALUES'][machine_mem_type]
    return [machine_cpu_cap, machine_mem_cap]
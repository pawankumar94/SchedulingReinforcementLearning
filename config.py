GYM_ENV_CFG = {
    'NB_NODES': 8,
    'SEED': 42,
    'cpu_req': 'uniform',
    'mem_req': 'medium',
    'TASK_LOAD': 'MEDIUM',
    'MC_CAP': 'BBACCBBBA',
    'MM_CAP': 'CCBBBBACA',
    'NB_RES_DIM': 2,
}
# MC_CAP_VALUES (Machine CPU Capacity values) is a representation of
# the total cores available per machine type
# MM_CAP_VALUES (Machine Memeory) represents the total normalized
# memory available per machine type
GLOBAL_CFG = {
    'Max_No_of_Task': 100,
    'Max_No_of_Jobs': 1000,
    'TASK_DURS_LOW': [50, 100],
    'TASK_DURS_MEDIUM': [50, 100, 200, 500],
    'TASK_DURS_HIGH': [200, 500, 1000],
    'MC_CAP_VALUES': {'A':0.25, 'B':0.5,'C':1.0},
    'MM_CAP_VALUES': {'A':0.25,'B':0.5,'C':0.75},
    'SEED': 42,
    'features_to_include': ['cpu_req', 'mem_req'],
    'DATA_LOCATION': 'input_1'
}

DRL_CFG = {
    'BATCH_SIZE': 64,
    'epsilon_start': 1.0,
    'epsilon_final': 0.01,
    'epsilon_decay': 500,
    'BETA_START': 0.4,
    'GAMMA': 0.99,
    'LR': 1e-3,
    'TARGET_UPD_INT': 1000,
    'MODEL_ARCH': 'dqn',
    'BUFFER_TYPE': 'PER',
    'OUTPUT_NODES': GYM_ENV_CFG['NB_NODES']
}



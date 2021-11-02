GYM_ENV_CFG = {
    # Machine Configuration: {5,8,10,20}
    'NB_NODES': 8,
    'SEED': 42,
    'cpu_req': 'uniform',
    'mem_req': 'uniform',
    'TASK_LOAD': 'MEDIUM',
    'MC_CAP': 'BBBBBBBA',
    'MM_CAP': 'BBBBBBBA',
    'NB_RES_DIM': 2,
    'One-Task': True,
    #[100, 200, 400, 500]
    "Total_Episodes": 200,
    #[True, False]
    "Episode_copy" : True
}
# MC_CAP_VALUES (Machine CPU Capacity values) is a representation of
# the total cores available per machine type
# MM_CAP_VALUES (Machine Memory) represents the total normalized
# memory available per machine type

GLOBAL_CFG = {
    # [1_000, 10_000, 100_000]
    'Max_No_of_Task': 1_000,
    'Max_No_of_Jobs': 10,
    'TASK_DURS_LOW': [50, 100],
    'TASK_DURS_MEDIUM': [50, 100, 200, 500],
    'TASK_DURS_HIGH': [200, 500, 1000],
    'MC_CAP_VALUES': {'A':1.0, 'B':0.50},
    'MM_CAP_VALUES': {'A': 0.75, 'B': 0.38},
    'SEED': 42,
    'features_to_include': ['cpu_req', 'mem_req'],
    'K_u': 2,
    'Scale_factor': 10,
    'Usage_Ratio': {"Cpu": 0.4,'Mem': 0.5},
    "Input_path" : "/nfs/home/kumarp/Final_100_files_data/Preprocessed_final_data.csv"
}

DRL_CFG = {
    'BATCH_SIZE': 32,
    'epsilon_start': 1.0,
    'min_mem_size': 35,
    'epsilon_final': 0.001,
    'replay_size': 1_00_000,
    'epsilon_decay': 0.9975,
    'BETA_START': 0.4,
    'GAMMA': 0.99,
    'LR': 1e-3,
    'TARGET_UPD_INT': 500,
    # [DQN, DDQN,DUELLINGDQN]
    'MODEL_ARCH': 'DUELLINGDQN',
    #['NORMAL','PER']
    'BUFFER_TYPE': 'PER',
    'OUTPUT_NODES': GYM_ENV_CFG['NB_NODES'],
    #[1,3,5,10]
    'Update_target_every': 2,
    # ['under_util', 'over_util' , 'simple']
    'reward_type': 'under_util',
    # [RMSPROP, ADAM]
    'Optimizer': "ADAM",
    # [HUBER, TDERROR]
    'Loss': "TDERROR",
    "Gradien_Clip_Val": 1.0
}



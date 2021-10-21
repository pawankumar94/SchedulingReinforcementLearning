import copy

from dqn_code import *
import matplotlib.pyplot as plt
env = make_data_set()
q_model = DQN(env)
target_model = DQN(env)

if DRL_CFG['BUFFER_TYPE'] == 'Normal':
    replay_buffer = Replay_Buffer(capacity=1_00_000)
elif DRL_CFG['BUFFER_TYPE'] == 'PER':
    replay_buffer = NaivePrioritizedBuffer(capacity=1_00_00)

total_episodes = len(env.all_episodes_duration)
epsilon = DRL_CFG['epsilon_start']
batch_size = DRL_CFG['BATCH_SIZE']

losses = []
rewards = []

for episode in range(total_episodes):
    state = env.reset()
    done = False
    timestep = 0
    total_timestep_epi = len(env.all_episodes_duration[episode])
    while not done:
        timestep += 1
        action = q_model.select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        epsilon = decay_epsilon(epsilon)
        print(timestep)
        state = copy.copy(next_state)

        if replay_buffer.len() > batch_size:
            loss = compute_td_loss(replay_buffer=replay_buffer, current_model=q_model,\
                            target_model=target_model, batch_size=batch_size)
            losses.append(loss)

        if done:
            update_target(q_model, target_model)
            plt.plot(losses)
            break

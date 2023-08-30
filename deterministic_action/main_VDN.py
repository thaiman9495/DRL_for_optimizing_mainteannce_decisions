import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque
from my_package.agent import VDN_Agent
from my_package.environment import System_Deterministic_IM

# mode = 0 --> training mode
# mode = 1 --> evaluation mode
mode = 0

n_steps_training = 1000000                 # Number of training steps
buffer_capacity = 200000                  # Size of relay buffer
buffer_batch_size = 128                    # Size of mini-batch sampled from replay buffer
gamma = 0.99                               # Discount factor
lr = 0.008                                  # Learning rate
lr_step_size = 20000                      # Period of learning rate decay
lr_gamma = 0.5                             # Multiplicative factor of learning rate decay
lr_min = 0.00025                           # Minimal learning rate
epsilon = 0.05                             # Constant for exploration
hidden_shared = [128, 128]                 # Config for hidden layers in shared module
hidden_value = [128]                       # Config for hidden layers in value module
hidden_advantage = [64]                   # Config for hidden layers in advantage module
n_target = 5000                           # Constant for updating target network

reward_scale = 10.0

# Number of components
n_c = 4
data_path = f'C:/Users/thaim/Documents/data_holder/papers/first_paper/revision_2/' \
            f'{n_c}_component_systems/Deterministic_IM/VDN/policy/'

# Configure forcasting path
n_run = 1
path_parent = Path(os.getcwd()).parent
forcast_path = str(path_parent).replace('\\', '/') + f'/cost_forecasting/data_holder/run_{n_run}/'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
transition_matrices = torch.load(forcast_path + 'estimated_trainsion_matrices.pt')
cost_model = torch.load(forcast_path + 'cost_model.pt')

# Initialize environment
env = System_Deterministic_IM(transition_matrices, cost_model, device)

n_components = env.n_components
n_component_actions = 5

# Initialize VDN agent
agent = VDN_Agent(n_components, n_component_actions,
                  hidden_shared, hidden_value, hidden_advantage,
                  lr, lr_step_size, lr_gamma, lr_min, n_target,
                  buffer_batch_size, buffer_capacity, gamma, device)

# Training loop
cost_holder = deque(maxlen=5000)
list_cost_rate = []
list_step = []

starting_time = datetime.datetime.now()
if mode == 0:
    for counter_step in range(0, n_steps_training):
        if counter_step % 1000 == 0:
            env.reset()

        # Get current state (system's state before maintenance)
        state = env.get_state()

        # Choose an action using actor network
        action = agent.choose_action(state, epsilon)

        # Perform action
        next_state, cost = env.perform_action(action, is_action_index=False)

        # Put experience in the bufffer
        agent.update_buffer(state, action, -cost/reward_scale, next_state)

        # Update policy
        if counter_step % 1 == 0:
            agent.update_policy(counter_step)

        # Print log
        cost_holder.append(cost)

        if counter_step % 1000 == 0:
            cost_rate = sum(cost_holder) / len(cost_holder)
            list_step.append(counter_step)
            list_cost_rate.append(cost_rate)
            print(f'Step: {counter_step}, cost rate: {cost_rate: .2f}, epsilon: {epsilon: .4f},'
                  f' lr: {agent.lr_scheduler.get_last_lr()[-1]}')

            agent.save_policy(step=counter_step, path=data_path)

    ending_time = datetime.datetime.now()
    training_time = ending_time - starting_time
    print(f"Training time: {training_time}")

    torch.save(list_step, 'data_holder/VDN/log_step.pt')
    torch.save(list_cost_rate, 'data_holder/VDN/log_cost_rate_train.pt')
    torch.save(training_time, 'data_holder/VDN/training_time.pt')

    plt.plot(list_step, list_cost_rate)
    plt.show()

else:
    log_step = torch.load('data_holder/VDN/log_step.pt')

    agent.policy_net.eval()
    log_cost_rate = []
    n_runs = 1
    n_interactions = 5000
    for step in log_step:
        policy = torch.load(data_path + f'policy_{step}.pt')
        agent.policy_net.load_state_dict(policy)

        average_cost = np.zeros(n_runs)

        for i in range(n_runs):
            total_cost = 0.0
            env.reset()
            for _ in range(n_interactions):
                # Get current state (system's state before maintenance)
                state = env.get_state()

                # Choose an action using actor network
                action = agent.choose_action(state, 0.0)

                # Perform action
                _, cost = env.perform_action(action, is_action_index=False)

                total_cost += cost

            average_cost[i] = total_cost / n_interactions

        mean_cost_rate = np.mean(average_cost)
        log_cost_rate.append(mean_cost_rate)
        print(f'step: {step}, cost_rate: {mean_cost_rate: .3f}')

    torch.save(log_cost_rate, 'data_holder/VDN/log_cost_rate.pt')

    plt.plot(log_step, log_cost_rate)
    plt.show()


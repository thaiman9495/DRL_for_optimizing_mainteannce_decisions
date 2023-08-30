import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from my_package.agent import VDN_Agent
from my_package.environment import System_Random_IM_Ground_Truth, System_Deterministic_IM_Ground_Truth

policy_number_random = 3995000
policy_number_deterministic = 3850000

n_steps_training = 4000000                 # Number of training steps
buffer_capacity = 500000                   # Size of relay buffer
buffer_batch_size = 128                    # Size of mini-batch sampled from replay buffer
gamma = 0.99                               # Discount factor
lr = 0.008                                 # Learning rate
lr_step_size = 50000                       # Period of learning rate decay
lr_gamma = 0.5                             # Multiplicative factor of learning rate decay
lr_min = 0.00025                           # Minimal learning rate
epsilon = 0.05                             # Constant for exploration
hidden_shared = [256, 256]                 # Config for hidden layers in shared module
hidden_value = [128]                       # Config for hidden layers in value module
hidden_advantage = [128]                   # Config for hidden layers in advantage module
n_target = 10000                           # Constant for updating target network


n_c = 15
data_path_parent = 'C:/Users/thaim/Documents/data_holder/papers/first_paper/revision_2/'
data_path_random = data_path_parent + f'{n_c}_component_systems/Random_IM/VDN/policy/'
data_path_deterministic = data_path_parent + f'{n_c}_component_systems/Deterministic_IM/VDN/policy/'

# Set training device: GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Initialize environment
# env = System_Random_IM_Ground_Truth(0.0)
env = System_Deterministic_IM_Ground_Truth(0.0)
n_components = env.n_components

# Initialize VDN agent
agent_random = VDN_Agent(n_components, 3,
                         hidden_shared, hidden_value, hidden_advantage,
                         lr, lr_step_size, lr_gamma, lr_min, n_target,
                         buffer_batch_size, buffer_capacity, gamma, device)

agent_deterministic = VDN_Agent(n_components, 5,
                                hidden_shared, hidden_value, hidden_advantage,
                                lr, lr_step_size, lr_gamma, lr_min, n_target,
                                buffer_batch_size, buffer_capacity, gamma, device)

# Load trained policy
policy_radom = torch.load(data_path_random + f'policy_{policy_number_random}.pt')
policy_deterministic = torch.load(data_path_deterministic + f'policy_{policy_number_deterministic}.pt')
agent_random.policy_net.load_state_dict(policy_radom)
agent_deterministic.policy_net.load_state_dict(policy_deterministic)

env.reset()
for _ in range(500):
    # Get current state (system's state before maintenance)
    state = env.get_state()

    # Choose an action using actor network
    action_random = agent_random.choose_action(state, 0.0)
    action_deterministic = agent_deterministic.choose_action(state, 0.0)
    print(f'{state} --> {action_random} --> {action_deterministic}')

    # Perform action
    _, _ = env.perform_action(action_deterministic, is_action_index=False)


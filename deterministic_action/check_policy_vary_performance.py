import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from my_package.agent import Dueling_DDQN_Agent
from my_package.environment import System_Deterministic_IM_Ground_Truth

case = 2
policy_number = 2650000
n_runs = 10
n_interactions = 5000

n_steps_training = 500000                # Number of training steps
buffer_capacity = 500000                 # Size of relay buffer
buffer_batch_size = 128                  # Size of mini-batch sampled from replay buffer
gamma = 0.99                             # Discount factor
lr = 0.001                               # Initial learning rate
lr_step_size = 50000                     # Period of learning rate decay
lr_gamma = 0.5                           # Multiplicative factor of learning rate decay
lr_min = 0.00025                         #
epsilon = 0.1                            # Constant for exploration
hidden_shared = [256, 256]               # Config for hidden layers in shared module
hidden_value = [128]                     # Config for hidden layers in value module
hidden_advantage = [128]                 # Config for hidden layers in advantage module
n_target = 5000                          # Constant for updating target network


n_c = 5      # Number of components
data_path = f'C:/Users/thaim/Documents/data_holder/papers/first_paper/revision_1/' \
            f'{n_c}_component_systems/Deterministic_IM/Dueling_DDQN/vary_transition_matrices/case_{case}/policy/'

# Set training device: GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Initialize environment
env = System_Deterministic_IM_Ground_Truth(0.0)                # Environment with varied transition matrices

n_components = env.n_components
n_actions = env.n_actions

# Initialize DDPG agent
agent = Dueling_DDQN_Agent(n_components, n_actions,
                           hidden_shared, hidden_value, hidden_advantage,
                           lr, lr_step_size, lr_gamma, lr_min, n_target,
                           buffer_batch_size, buffer_capacity,
                           gamma, device)

# Load trained policy
policy = torch.load(data_path + f'policy_{policy_number}.pt')
agent.policy_net.load_state_dict(policy)

# Check
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
        _, cost = env.perform_action(action)

        total_cost += cost

    average_cost[i] = total_cost / n_interactions

mean_cost_rate = np.mean(average_cost)
print(mean_cost_rate)

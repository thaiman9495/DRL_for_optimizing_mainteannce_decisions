import os
import numpy as np
import torch

from pathlib import Path
from my_package.environment import System_Random_IM

n_runs = 20
n_interactions = 5000

# Configure forcasting path
n_run_forcast = 1
path_parent = Path(os.getcwd()).parent
forcast_path = str(path_parent).replace('\\', '/') + f'/cost_forecasting/data_holder/run_{n_run_forcast}/'

# Load policy
policy = torch.load('data_holder/VI/policy.pt')

for state, action in policy.items():
    print(f'{state} -----> {action}')

# Set training device: GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Load estimated transition matrices
transition_matrices = torch.load(forcast_path + 'estimated_trainsion_matrices.pt')

# Load trained cost model
cost_model = torch.load(forcast_path + 'cost_model.pt')

# Initialize environment
env = System_Random_IM(transition_matrices, cost_model, device)

average_cost = np.zeros(n_runs)

for i in range(n_runs):
    total_cost = 0.0
    env.reset()
    for _ in range(n_interactions):
        # Get current state
        state = env.get_state()

        # Choose action
        action = np.array(policy[tuple(state)], dtype=int)

        # Perform action
        _, cost = env.perform_action(action, is_action_index=False)

        total_cost += cost

    average_cost[i] = total_cost / n_interactions

mean_cost_rate = np.mean(average_cost)
std_cost_rate = np.std(average_cost)

torch.save(mean_cost_rate, 'data_holder/VI/cost_rate.pt')
print(mean_cost_rate)
print(std_cost_rate)





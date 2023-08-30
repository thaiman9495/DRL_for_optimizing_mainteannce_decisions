import os
import torch
import datetime
import numpy as np

from pathlib import Path
from itertools import product
from my_package.environment import System_Random_IM

# Number of interactions used to test policy
n_runs = 5
n_validation = 5000

# Configure forcasting path
n_run_forcast = 1
path_parent = Path(os.getcwd()).parent
forcast_path = str(path_parent).replace('\\', '/') + f'/cost_forecasting/data_holder/run_{n_run_forcast}/'

# Loading ...
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
transition_matrices = torch.load(forcast_path + 'estimated_trainsion_matrices.pt')
cost_model = torch.load(forcast_path + 'cost_model.pt')

# Initialize environment
env = System_Random_IM(transition_matrices, cost_model, device)
n_components = env.n_components
n_component_states = env.n_component_states

# Main loop
control_limit_space = np.array(tuple(product(range(1, n_component_states-1), repeat=n_components)), dtype=int)
average_cost_min = float('inf')
control_limits_min = np.zeros((n_components,), dtype=int)

starting_time = datetime.datetime.now()
for control_limit in control_limit_space:
    cost_rate = np.zeros(n_runs)

    for k in range(n_runs):
        env.reset()
        total_cost = 0.0
        for _ in range(n_validation):
            # Get current state
            state = env.get_state()

            # Choose action
            action = np.zeros((n_components,), dtype=int)
            for i in range(n_components):
                if state[i] < control_limit[i]:
                    action[i] = 0
                else:
                    if state[i] < n_component_states - 1:
                        action[i] = 1
                    else:
                        action[i] = 2

            # Perform action
            _, cost = env.perform_action(action, is_action_index=False)

            # Compute total cost
            total_cost += cost

        # Compute average cost
        cost_rate[k] = total_cost / n_validation

    average_cost = np.mean(cost_rate)
    print(f'{control_limit}: {average_cost}')

    if average_cost_min > average_cost:
        average_cost_min = average_cost
        control_limits_min = control_limit

ending_time = datetime.datetime.now()
training_time = ending_time - starting_time
print(f'Duration of one sweep: {training_time}')

print('')
print(f'Minimal average cost: {average_cost_min}')
print(f'Control limit:        {control_limits_min}')

torch.save(control_limits_min, 'data_holder/Threshold_based_policy/grid_search/control_limits.pt')
torch.save(average_cost_min, 'data_holder/Threshold_based_policy/grid_search/cost_rate.pt')
torch.save(training_time, 'data_holder/Threshold_based_policy/grid_search/training_time.pt')


import numpy as np
import pandas as pd
from my_package.environment import System_Data_Generation

system = System_Data_Generation()        # Load system
n_components = system.n_components       # Number of components
n_states = system.n_component_states     # Number of states of a component


n_inspection = 30000                                       # Dataset size
maintenance_action = np.zeros(n_components, dtype=int)    # Initilize maintenance action

# Dictionary for holding data
dataset_I_f = np.zeros((n_inspection,))                                # System status
dataset_state_bm = np.zeros((n_inspection, n_components), dtype=int)   # State before maintenance
dataset_state_am = np.zeros((n_inspection, n_components), dtype=int)   # State after maintenance
dataset_cost = np.zeros((n_inspection,))                               # Maintenance cost

for k in range(n_inspection):
    # Check first inspection
    if k == 0:
        is_first_inspection = True
    else:
        is_first_inspection = False

    # Get state before maintenance
    state_bm = system.get_state()

    # Choose action
    for i in range(n_components):
        maintenance_action[i] = np.random.randint(0, state_bm[i] + 1)

    I_f, state_am, cost = system.perform_action(maintenance_action, is_first_inspection)

    dataset_I_f[k] = I_f
    dataset_state_bm[k, :] = state_bm
    dataset_state_am[k, :] = state_am
    dataset_cost[k] = cost

# --------------------------------------------
# Save data under excel format
# --------------------------------------------

# Dictionary for holding all data
my_dict = {}

# Add system status to dataset
my_dict['System_status'] = dataset_I_f

# Add state before maintenance to my dictionary
for i in range(n_components):
    my_dict[f'C{i + 1}'] = dataset_state_bm[:, i]

# Add state after maintenance to my dictionary
for i in range(n_components):
    my_dict[f'MC{i + 1}'] = dataset_state_am[:, i]

# Add total maintenance cost to my dictionary
my_dict['Cost'] = dataset_cost

# Convert from dictionary to DataFrame
df_dataset = pd.DataFrame(my_dict)

# Save to excel files
df_dataset.to_excel("maintenance_data.xlsx", index=False)

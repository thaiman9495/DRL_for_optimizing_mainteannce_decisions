import torch
import numpy as np
import pandas as pd

# Preprocess dataset

n_components = 15                                             # Number of components
n_states_per_component = 5                                   # Number of states per component

dataset = pd.read_excel("maintenance_data.xlsx")             # Load dataset
data = dataset.drop(['System_status'], axis=1).to_numpy()    # Convert from DataFrame to Numpy
data_bm = data[:, 0:n_components]                            # Dataset for state before maintenance
data_am = data[:, n_components:2*n_components]               # Dataset for state after maintenance
n_samples = len(data_am)                                     # Number of samples in dataset

# Initialize transition matrices
transition_matrices = np.zeros((n_components, n_states_per_component, n_states_per_component))
for i in range(n_components):
    transition_matrices[i, n_states_per_component-1, n_states_per_component-1] = 1.0

# Compute transition matrices
for sample in range(n_samples-1):
    state_am = data_am[sample, :]
    state_bm = data_bm[sample+1, :]
    for i in range(n_components):
        for j in range(n_states_per_component):
            if state_am[i] == j:
                for k in range(n_states_per_component):
                    if state_bm[i] == k:
                        transition_matrices[i, j, k] += 1

for i in range(n_components):
    row_sum = transition_matrices[i, :, :].sum(axis=1)
    for j in range(n_states_per_component):
        transition_matrices[i, j, :] = transition_matrices[i, j, :]/row_sum[j]

# Save estmated transition matrices
torch.save(transition_matrices, 'estimated_trainsion_matrices.pt')

# Print transition matrices
for i in range(n_components):
    print(f"Transition matrix for component {i+1}")
    print(transition_matrices[i, :, :])
    print("")






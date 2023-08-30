import os
import torch
import datetime
import numpy as np

from pathlib import Path
from itertools import product
from my_package.environment import System_Deterministic_IM

theta = 0.001      # presison
gamma = 0.95       # discount factor

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
n_component_states = env.n_component_states

# ----------------------------------------------------------------------------------------------------------------------
# Create state space
# ----------------------------------------------------------------------------------------------------------------------
index_to_state = np.array(tuple(product(range(n_component_states), repeat=n_components)), dtype=int)
state_to_index = np.zeros(tuple(n_component_states for _ in range(n_components)), dtype=int)
for idx, state in enumerate(index_to_state):
    state_to_index[tuple(state)] = idx

n_system_states = len(index_to_state)

# ----------------------------------------------------------------------------------------------------------------------
# Create action space
# ----------------------------------------------------------------------------------------------------------------------
index_to_action = index_to_state
action_to_index = state_to_index
n_actions = len(index_to_action)

# ----------------------------------------------------------------------------------------------------------------------
# Create a dictionnary of fesible actions for each state
# ----------------------------------------------------------------------------------------------------------------------
feasible_action = {}
for i in range(n_system_states):
    state = index_to_state[i]
    temp = tuple(range(value + 1) for value in state)
    list_actions = np.array(tuple(product(*temp)), dtype=int)
    feasible_action[i] = list_actions

# ----------------------------------------------------------------------------------------------------------------------
# Create a dictionnary of fesible states after degradation for each state
# ----------------------------------------------------------------------------------------------------------------------
feasible_next_state_index = {}
for i in range(n_system_states):
    state = index_to_state[i]
    temp = tuple(range(value, n_component_states) for value in state)
    list_states = np.array(tuple(product(*temp)), dtype=int)
    list_state_index = [state_to_index[tuple(state)] for state in list_states]
    feasible_next_state_index[i] = list_state_index

# Compute system transition probability
p_system = np.zeros((n_system_states, n_system_states))
for i in range(n_system_states):
    state = index_to_state[i, :]
    for j in range(n_system_states):
        next_state = index_to_state[j, :]
        probability = 1.0
        for k in range(n_components):
            probability *= transition_matrices[k, state[k], next_state[k]]

        p_system[i, j] = probability

# ----------------------------------------------------------------------------------------------------------------------
# Main training part
# ----------------------------------------------------------------------------------------------------------------------
V = - np.random.randn(n_system_states)   # Initalize value function
delta = 1.0

print('Start sweeping on state space')
starting_time = datetime.datetime.now()

while delta > theta:
    delta = 0.0
    for i in range(n_system_states):
        state = index_to_state[i]
        # Save old version of value function for comparison purpose
        V_old = V[i]

        # Get feasible action for the current state
        list_feasible_action = feasible_action[i]

        # Iterate over all possile actions to update value function
        V_max = - float('inf')

        for a in list_feasible_action:
            # Convert from tuple to numpy
            action = np.array(a, dtype=int)

            # Compute state after maintenance
            state_am = state - action

            # Get states at next time step
            state_am_index = state_to_index[tuple(state_am)]
            list_feasible_next_state_index = feasible_next_state_index[state_am_index]

            # Compute updates
            update = 0.0
            reward = env.reward_function(state, action)

            for next_state_index in list_feasible_next_state_index:
                probability = p_system[state_am_index, next_state_index]
                update += probability * (reward + gamma * V[next_state_index])

            V_max = max(V_max, update)

        V[i] = V_max
        delta = max(delta, abs(V_old - V[i]))

    print(f'delta: {delta: .10f}, theta: {theta: .5f}')

# Get optimal policy
optimal_policy = {}

for i in range(n_system_states):
    state = index_to_state[i]

    # Get feasible action for the current state
    list_feasible_action = feasible_action[i]

    V_max = - float('inf')
    a_max = index_to_action[0]

    for a in list_feasible_action:
        # Convert from tuple to numpy
        action = np.array(a)

        # Compute state after maintenance
        state_am = state - action

        # Get states at next time step
        state_am_index = state_to_index[tuple(state_am)]
        list_feasible_next_state_index = feasible_next_state_index[state_am_index]

        # Compute updates
        update = 0.0
        reward = env.reward_function(state, action)

        for next_state_index in list_feasible_next_state_index:
            probability = p_system[state_am_index, next_state_index]
            update += probability * (reward + gamma * V[next_state_index])

        if update > V_max:
            V_max = update
            a_max = a

    optimal_policy[tuple(state)] = a_max

ending_time = datetime.datetime.now()
training_time = ending_time - starting_time
print(f'Duration of one sweep: {training_time}')

torch.save(optimal_policy, 'data_holder/VI/policy.pt')
torch.save(training_time, 'data_holder/VI/training_time.pt')











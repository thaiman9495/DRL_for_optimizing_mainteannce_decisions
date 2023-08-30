import os
import torch

from pathlib import Path
from my_package.environment import System_Random_IM, System_Random_IM_Ground_Truth
from my_package.agent import Generic_Algorithm

# Hyper-parameters
n_iterations = 50
population_size = 10
crossover_prob = 0.8
n_interventions = 10000
mutation_prob = 0.2
n_runs = 5

# Configure forcasting path
# n_run_forcast = 1
path_parent = Path(os.getcwd()).parent
# forcast_path = str(path_parent).replace('\\', '/') + f'/cost_forecasting/data_holder/run_{n_run_forcast}/'
#
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# transition_matrices = torch.load(forcast_path + 'estimated_trainsion_matrices.pt')
# cost_model = torch.load(forcast_path + 'cost_model.pt')

# Initialize environment
# env = System_Random_IM(transition_matrices, cost_model, device)
env = System_Random_IM_Ground_Truth(0.0)
n_components = env.n_components
n_component_states = env.n_component_states

# Initilize ga agent
agent = Generic_Algorithm(n_components, n_component_states, n_iterations, population_size,
                          crossover_prob, mutation_prob, n_interventions, n_runs)

# Do it
path_log = str(path_parent).replace('\\', '/') + '/random_action/data_holder/Threshold_based_policy/ga/'
agent.train(env, path_log)




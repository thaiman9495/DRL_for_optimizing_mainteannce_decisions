import numpy as np
import torch

from my_package.environment import System_Data_Generation

# Set training device: GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("This program is trained on GPU\n")
else:
    device = torch.device("cpu")
    print("This program is trained on CPU\n")

# Load system
system = System_Data_Generation()
n_components = system.n_components

# Load cost model
# cost_model = torch.load('data_holder/run_1/cost_model.pt')
cost_model = torch.load('data_holder/cost_model.pt')

n_inspection = 100
maintenance_action = np.zeros(n_components, dtype=np.int32)

for k in range(n_inspection):
    # Get state before maintenance
    state_bm = system.get_state()

    # Choose action
    for i in range(n_components):
        maintenance_action[i] = np.random.randint(0, state_bm[i] + 1)

    if k == 0:
        _, state_am, cost_true = system.perform_action(maintenance_action, is_first_step=True)
    else:
        _, state_am, cost_true = system.perform_action(maintenance_action, is_first_step=False)

    my_input = torch.tensor(np.append(state_bm, state_am), dtype=torch.float32).to(device)
    cost_estimate = cost_model(my_input).item()

    print(f' {state_bm}, {state_am}, {round(cost_estimate, 2)}, {round(cost_true, 2)}')


# # Load  model
# state_bm = torch.tensor([1, 1, 0, 2, 0], dtype=torch.float32).to(device)
# state_am = torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32).to(device)
# my_input = torch.atleast_2d(torch.cat([state_bm, state_am]))
#
# cost_model = torch.load('cost_model.pt').eval()
# cost = cost_model(my_input)
# print(cost)




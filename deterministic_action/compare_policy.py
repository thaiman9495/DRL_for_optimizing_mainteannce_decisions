import numpy as np
import torch
import matplotlib.pyplot as plt

# VDN
step_vdn = torch.load('data_holder/VDN/log_step.pt')
cost_rate_vdn = torch.load('data_holder/VDN/log_cost_rate.pt')

# VI
n_steps = len(step_vdn)
step_vi = step_vdn
cost_rate_vi = np.ones(n_steps) * torch.load(f'data_holder/VI/cost_rate.pt')

plt.plot(step_vi, cost_rate_vi, label='Value Iteration (exact solution)')
plt.plot(step_vdn, cost_rate_vdn, label='VDN')

plt.legend()
plt.show()





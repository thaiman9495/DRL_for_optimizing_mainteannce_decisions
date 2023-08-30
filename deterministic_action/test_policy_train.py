import numpy as np
import torch
import matplotlib.pyplot as plt


# Dueling DDQN
step_dueling_ddqn = torch.load('data_holder/Dueling_DDQN/log_step.pt')
cost_rate_dueling_ddqn = torch.load('data_holder/Dueling_DDQN/log_cost_rate_train.pt')

plt.plot(step_dueling_ddqn, cost_rate_dueling_ddqn, label='Dueling DDQN')
plt.legend()
plt.show()





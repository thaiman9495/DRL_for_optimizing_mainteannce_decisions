import torch
import matplotlib.pyplot as plt

step = torch.load(f'data_holder/VDN/log_step.pt')
cost = torch.load(f'data_holder/VDN/log_cost_rate.pt')
training_time = torch.load(f'data_holder/VDN/training_time.pt')

print(training_time)

plt.plot(step, cost)
plt.show()



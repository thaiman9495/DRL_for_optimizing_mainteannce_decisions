import torch
import matplotlib.pyplot as plt

plt.rc('font', size=20)                       # Change font size
plt.rc('font', family='serif')                # Change font
plt.rc('lines', linewidth=1.5)                # Change line width
plt.rc('text', usetex=True)                   # Use Latex


def format_func(value, tick_number):
    N = value / 1000000
    return f"{N}"


step = torch.load('random_action/data_holder/VDN/log_step.pt')
cost = torch.load('random_action/data_holder/VDN/log_cost_rate.pt')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(step, cost, linestyle='-', linewidth=2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.set_xlabel(r'Step ($\times 10^6$)')
ax.set_ylabel('Average cost rate')

plt.show()

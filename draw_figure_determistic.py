import torch
import matplotlib.pyplot as plt

plt.rc('font', size=20)                       # Change font size
plt.rc('font', family='serif')                # Change font
plt.rc('lines', linewidth=1.5)                # Change line width
plt.rc('text', usetex=True)                   # Use Latex


def format_func(value, tick_number):
    N = value / 1000000
    return f"{N}"


log_step = torch.load('deterministic_action/data_holder/VDN/vary_transition_matrices/log_step.pt')
log_cost_rate = torch.load('deterministic_action/data_holder/VDN/vary_transition_matrices/log_cost_rate.pt')

# log_step_reward_shaping = torch.load('deterministic_action/data_holder/VDN/log_step_shaping_reward.pt')
# log_cost_rate_reward_shaping = torch.load('deterministic_action/data_holder/VDN/log_cost_rate_shaping_reward.pt')

step = log_step[1:]
cost_rate = log_cost_rate[1:]
# step_reward_shaping = log_step_reward_shaping[1:]
# cost_rate_reward_shaping = log_cost_rate_reward_shaping[1:]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(step, cost_rate, linestyle='-', linewidth=2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.set_xlabel(r'Step ($\times 10^6$)')
ax.set_ylabel('Average cost rate')

plt.show()


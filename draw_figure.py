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
log_cost_rate_determinis = torch.load('deterministic_action/data_holder/VDN/vary_transition_matrices/log_cost_rate.pt')
log_cost_rate_random = torch.load('random_action/data_holder/VDN/vary_transition_matrices/log_cost_rate.pt')

# log_step_reward_shaping = torch.load('deterministic_action/data_holder/VDN/log_step_shaping_reward.pt')
# log_cost_rate_reward_shaping = torch.load('deterministic_action/data_holder/VDN/log_cost_rate_shaping_reward.pt')

step = log_step[0:]
cost_rate_deterministic = log_cost_rate_determinis[0:]
cost_rate_random = log_cost_rate_random[0:]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(step, cost_rate_deterministic, linestyle='-', linewidth=2, label='Deterministic IM')
ax.plot(step, cost_rate_random, linestyle='-', linewidth=2, label='Random IM')
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.set_xlabel(r'Step ($\times 10^6$)')
ax.set_ylabel('Cost rate')

plt.legend()
plt.show()


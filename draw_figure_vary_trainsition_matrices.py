import matplotlib.pyplot as plt

plt.rc('font', size=17)                       # Change font size
plt.rc('font', family='serif')                # Change font
plt.rc('lines', linewidth=1.5)                # Change line width
plt.rc('text', usetex=True)                   # Use Latex


alpha = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
percentage_random = [0.0, 0.20, 0.23, 1.11, 1.21, 1.44, 1.48, 1.60, 1.83, 2.60, 4.04]
percentage_determistic = [0.0, 0.40, 0.55, 1.32, 1.85, 2.09, 2.75, 3.05, 3.72, 5.82, 7.16]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(alpha, percentage_random, marker='o', linestyle='-', linewidth=2,
        label='Random imperfect maintenance')
ax.plot(alpha, percentage_determistic, marker='o', linestyle='-', linewidth=2,
        label='Deterministic imperfect maintenance')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$CP (\%)$')
plt.legend()
plt.show()

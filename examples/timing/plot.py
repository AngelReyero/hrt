import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



experiment = 'basics'
model = 'ols'
N = 200
P = 50
nruns = 50
nfolds = 5
ntested = 50
nsignals = 3
response_structure = 'tanh'


if experiment == 'cv':
    testers = ['CV-HRT', 'Invalid CV-HRT','CV-CPI','Invalid CV-CPI','Invalid binom-HRT','CV-HGT','CV-HPT']
elif experiment == 'sample_size':
    test_percentage = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    testers = [f'binom{x}' for x in test_percentage]
elif experiment == 'ntrials':
    ntrials = [1, 5, 10, 30, 50, 100, 500, 10000]
    testers = [f'binom{x}' for x in ntrials]
elif experiment == 'basics':
    testers = ['Basic HRT','Basic HGT','Basic HPT','Basic Binom_CRT','Basic CPI','Invalid CPI']


ntesters = len(testers)

# Load CSV
loaded_data = np.loadtxt(f"csv/{experiment}/N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.csv", delimiter=",", skiprows=1)

# Split back into p_values and timings
p_values = loaded_data[:, 0].reshape((nruns, ntested, ntesters))
timings = loaded_data[:, 1].reshape((nruns, ntested, ntesters))






grid = np.linspace(0,1,1000)

# Colorblind-friendly color cycle
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00',
                'black']
fig = plt.figure()
for tester_idx, (label, color) in enumerate(zip(testers, CB_color_cycle)):
    cdf_signals = (p_values[:,:nsignals,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
    cdf_nulls = (p_values[:,nsignals:,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
    plt.plot(grid, cdf_signals, label=label, ls='-', color=color)
    plt.plot(grid, cdf_nulls, ls=':', color=color)
plt.plot([0,1],[0,1], color='black')

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.65)   

# plt.legend(loc='lower right', ncol=2)
plt.savefig(f'plots/{experiment}/p_values_N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.pdf', bbox_inches='tight')
plt.close()

grid = np.linspace(0,0.05,1000)
fig = plt.figure()
for tester_idx, (label, color) in enumerate(zip(testers, CB_color_cycle)):
    cdf_signals = (p_values[:,:nsignals,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
    cdf_nulls = (p_values[:,nsignals:,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
    plt.plot(grid, cdf_signals, label=label, ls='-', color=color)
    plt.plot(grid, cdf_nulls, ls=':', color=color)
plt.plot([0,0.05],[0,0.05], color='black')

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.65)   

# plt.legend(loc='lower right', ncol=2)
plt.savefig(f'plots/{experiment}/zoomed_N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.pdf', bbox_inches='tight')
plt.close()




# TIME
# Reshape: collect all timing values per tester
data_for_boxplot = [
    timings[:, :, i].flatten() for i in range(ntesters)
]

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(data_for_boxplot, tick_labels=testers)
plt.yscale("log")
plt.xlabel("Methods")
plt.ylabel("Seconds")
plt.title("Computational cost")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f'plots/{experiment}/Time_N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.pdf', bbox_inches='tight')

       

# FDP and TPR
def compute_tpr_type_I(p_values, alpha, nsignal):
    nruns, ntested, ntesters = p_values.shape
    tprs = [[] for _ in range(ntesters)]
    type_Is = [[] for _ in range(ntesters)]

    for run in range(nruns):
        for tester in range(ntesters):
            p = p_values[run, :, tester]

            # Discoveries: p < alpha
            discoveries = np.where(p < alpha)[0]

            # True discoveries = indices < nsignal
            true_discoveries = discoveries[discoveries < nsignal]
            false_discoveries = discoveries[discoveries >= nsignal]

            # TPR = true positives / total true signals
            tpr = len(true_discoveries) / nsignal if nsignal > 0 else 0
            # FDP = false positives / total discoveries
            type_I = len(false_discoveries) / (ntested-nsignal) if len(discoveries) > 0 else 0

            tprs[tester].append(tpr)
            type_Is[tester].append(type_I)

    return tprs, type_Is


alpha = 0.05

tprs, type_I = compute_tpr_type_I(p_values, alpha, nsignals)

positions = np.arange(ntesters) * 2  # space between tester groups

fig, ax = plt.subplots(figsize=(12, 6))

# Offsets to align side-by-side boxes
width = 0.3

# Boxplots for TPR (green)
bp1 = ax.boxplot(tprs,
                 positions=positions - width,
                 widths=0.25,
                 patch_artist=True,
                 boxprops=dict(facecolor='green'),
                 medianprops=dict(color='black'))

# Boxplots for FDP (red)
bp2 = ax.boxplot(type_I,
                 positions=positions + width,
                 widths=0.25,
                 patch_artist=True,
                 boxprops=dict(facecolor='red'),
                 medianprops=dict(color='black'))

# X-ticks in center of each group
ax.set_xticks(positions)
ax.set_xticklabels(testers, rotation=45, ha="right")
ax.axhline(y=alpha, color='black', linestyle='--', linewidth=1, label=f'alpha = {alpha}')
ax.set_ylabel("Proportion")
ax.set_title(f"TPR (green) and type_I (red) per Tester (alpha={alpha})")
ax.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f'plots/{experiment}/type_N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.pdf', bbox_inches='tight')


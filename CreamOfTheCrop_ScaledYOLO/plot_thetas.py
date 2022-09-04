from pathlib import Path
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=15)


experiment_name = 'VOC-NAS-L' #'VOC-NAS-SS-03'
experiment_path = Path(f'experiments/workspace/train/{experiment_name}/')
thetas_path = experiment_path / 'thetas.txt'
experiment_graphs = Path('.')

thetas_string = []
with open(thetas_path, 'r') as file:
    for line in file:
        parsed_string = line.replace('array', '').replace(' ', '').replace('dtype=float32', '').replace(',),(', ',').replace('(', '').replace(',)', '').strip()
        thetas_string.append(json.loads(parsed_string))

fig, axs = plt.subplots(4, 2)
# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')
last_thetas = thetas_string[-1]
overall_plots = 0
for idx in range(len(last_thetas) // 2):
    for sub_plot in range(2):
        x = np.arange(4)
        ticks = range(12)
        width = 0.2
        axs[idx, sub_plot].bar(x - 0.5, height=last_thetas[overall_plots][:4], width=width, color=['r', 'r', 'r', 'r'], align='edge')
        axs[idx, sub_plot].bar(x - 0.3, height=last_thetas[overall_plots][4:8], width=width, color=['b', 'b', 'b', 'b'], align='edge')
        axs[idx, sub_plot].bar(x, height=last_thetas[overall_plots][8:12], width=width, color=['g', 'g', 'g', 'g'])
        axs[idx, sub_plot].set_xticks(x - width)
        axs[idx, sub_plot].set_xticklabels(['0', '6', '4', '2'])
        axs[idx, sub_plot].set_xlabel('Depth')
        axs[idx, sub_plot].set_ylabel('Softmax output')
        axs[idx, sub_plot].legend(['gamma=0.25', 'gamma=0.5', 'gamma=0.75'])
        overall_plots += 1
fig.set_size_inches(15.5, 13.5)
fig.tight_layout()
fig.savefig(experiment_path / 'thetas.png')
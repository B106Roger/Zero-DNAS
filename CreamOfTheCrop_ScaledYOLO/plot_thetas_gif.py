from pathlib import Path
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import imageio

from tqdm import tqdm
matplotlib.rc('font', size=15)


experiment_name = 'VOC-NAS-SS-S44-Beta-0025' #'VOC-NAS-SS-03'
search_space = ['0', '6', '4', '2']
experiment_path = Path(f'experiments/workspace/train/{experiment_name}/')
thetas_path = experiment_path / 'history_thetas.txt'
thetas_gif_path = experiment_path / 'history_thetas.gif'
experiment_graphs = Path('.')



with imageio.get_writer(thetas_gif_path, mode='I', fps=10) as writer:
    with open(thetas_path, 'r') as file:
        fig, axs = plt.subplots(4, 2)
        lines=file.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            parsed_string = line.replace('array', '').replace(' ', '').replace('dtype=float32', '').replace(',),(', ',').replace('(', '').replace(',)', '').strip()
            thetas_string = json.loads(parsed_string)    
            last_thetas = thetas_string
            overall_plots = 0
            for idx in range(len(last_thetas) // 2):
                for sub_plot in range(2):
                    axs[idx, sub_plot].clear()
                    x = np.arange(4)
                    ticks = range(12)
                    width = 0.2
                    axs[idx, sub_plot].bar(x - 0.5, height=last_thetas[overall_plots][:4], width=width, color=['r', 'r', 'r', 'r'], align='edge')
                    axs[idx, sub_plot].bar(x - 0.3, height=last_thetas[overall_plots][4:8], width=width, color=['b', 'b', 'b', 'b'], align='edge')
                    axs[idx, sub_plot].bar(x,       height=last_thetas[overall_plots][8:12], width=width, color=['g', 'g', 'g', 'g'])
                    axs[idx, sub_plot].set_xticks(x - width)
                    axs[idx, sub_plot].set_ylim([0.0, 1.0])
                    axs[idx, sub_plot].set_xticklabels(search_space)
                    axs[idx, sub_plot].set_xlabel('Depth')
                    axs[idx, sub_plot].set_ylabel('Softmax output')
                    axs[idx, sub_plot].legend(['gamma=0.25', 'gamma=0.5', 'gamma=0.75'])
                    overall_plots += 1

            plt.suptitle(str(i))
            fig.set_size_inches(12.4, 10.8)
            # fig.set_size_inches(9.3, 8.1)
            # fig.set_size_inches(6.2, 5.4)
            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(data)
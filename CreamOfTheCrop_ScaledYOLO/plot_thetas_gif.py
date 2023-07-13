from pathlib import Path
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import imageio
import argparse

from tqdm import tqdm
matplotlib.rc('font', size=15)

parser = argparse.ArgumentParser(description='Plot the theta distribution according to the serach result')
parser.add_argument('--exp_name', type=str, required=True)
args = parser.parse_args()

experiment_name = args.exp_name #'VOC-NAS-SS-03'
depth_search_space = ['0','1','2'] #['0','6','4','2']
gamma_search_space = ['0.25', '0.50', '0.75']

experiment_path = Path(f'experiments/workspace/train/{experiment_name}/')
thetas_path = experiment_path / 'history_thetas.txt'
thetas_gif_path = experiment_path / 'history_thetas.gif'
experiment_graphs = Path('.')

depth_len=len(depth_search_space)
gamma_len=len(gamma_search_space)

with imageio.get_writer(thetas_gif_path, mode='I', fps=2) as writer:
    with open(thetas_path, 'r') as file:
        fig, axs = plt.subplots(4, 2)
        lines=file.readlines()
        for line_id, line in tqdm(enumerate(lines), total=len(lines)):
            parsed_string = line.replace('array', '').replace(' ', '').replace('dtype=float32', '').replace(',),(', ',').replace('(', '').replace(',)', '').strip()
            thetas_string = json.loads(parsed_string)    
            last_thetas = thetas_string
            overall_plots = 0
            for idx in range(len(last_thetas) // 2):
                for sub_plot in range(2):
                    axs[idx, sub_plot].clear()
                    x = np.arange(depth_len)
                    ticks = range(depth_len*gamma_len)
                    width = 0.2
                    for i, (gamma, color) in enumerate(zip(gamma_search_space,['r','b','g'])):
                        axs[idx, sub_plot].bar(x - 1./gamma_len*i, height=last_thetas[overall_plots][i*depth_len:(i+1)*depth_len],     width=width, color=[color]*depth_len, align='edge')
                    axs[idx, sub_plot].set_xticks(x - width)
                    axs[idx, sub_plot].set_xticklabels(depth_search_space)
                    axs[idx, sub_plot].set_xlabel('Depth')
                    axs[idx, sub_plot].set_ylabel('Softmax output')
                    axs[idx, sub_plot].legend([f'gamma={gamma}' for gamma in gamma_search_space])
                    overall_plots += 1

            plt.suptitle(str(line_id))
            fig.set_size_inches(12.4, 10.8)
            # fig.set_size_inches(9.3, 8.1)
            # fig.set_size_inches(6.2, 5.4)
            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(data)
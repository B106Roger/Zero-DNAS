from pathlib import Path

import matplotlib.pyplot as plt

experiments_path = Path('experiments') / 'workspace' / 'train'
experiment_graphs = Path('experiment_results')

log_path_baseline = experiments_path / 'Baseline' / 'train.log'
log_path_attentive = experiments_path / 'MinLossFirstAttentive' / 'train.log'
log_path_attentive_synflow = experiments_path / 'MaxSynflowFirstAttentive' / 'train.log'


def parse_results(results_file):
    avg_flops = []
    current_epoch_sum = 0
    for line in results_file:
        if 'No' in line:
            flops = float(line.split('(')[1].split(',')[2].strip())
            current_epoch_sum += flops

            if 'No.9' in line:
                avg_flops.append(current_epoch_sum / 10)
                current_epoch_sum = 0

    return avg_flops

def plot_results(experiments, result_type, legend_position='lower right'):
    for experiment in experiments:
        plt.plot(range(len(experiment)), experiment)
    
    plt.xlabel('Epoch')
    plt.ylabel('Best candidate pool Avg. FLOPs')
    plt.legend(['Uniform Sampling', 'Attentive Sampling', 'Attentive Sampling(synflow)'], loc=legend_position)
    plt.savefig(experiment_graphs / f'{result_type}.png')
    plt.close()

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

with open(log_path_baseline, 'r') as baseline_results:
    avg_flops_baseline = parse_results(baseline_results)

with open(log_path_attentive, 'r') as baseline_results:
    avg_flops_attentive = parse_results(baseline_results)

with open(log_path_attentive_synflow, 'r') as baseline_results:
    avg_flops_attentive_synflow = parse_results(baseline_results)



plot_results(
    [smooth(avg_flops_baseline, 0.9), smooth(avg_flops_attentive, 0.9), smooth(avg_flops_attentive_synflow[20:], 0.9)], 
    'average flops',
    'upper right')
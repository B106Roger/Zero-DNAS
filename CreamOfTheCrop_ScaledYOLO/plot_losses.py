from pathlib import Path

import matplotlib.pyplot as plt

experiments_path = Path('experiments') / 'workspace' / 'train'
experiment_graphs = Path('experiment_results')

log_path_baseline = experiments_path / 'FirstHardwareMeta' / 'train.log'
log_path_attentive = experiments_path / 'MinLossFirstAttentive' / 'train.log'
log_path_attentive_synflow = experiments_path / 'MaxSynflowFirstAttentive' / 'train.log'


def parse_results(results_file):
    synflow_losses = []
    flops_losses = []
    for line in results_file:
        if 'Train:' in line:
            iteration_synflow = float(line.split('(')[1].split(',')[0].split('Loss:')[-1].strip())
            iteration_flops = float(line.split('Flops_loss')[1].split('(')[0].split(':')[-1].strip())
            synflow_losses.append(iteration_synflow)
            flops_losses.append(iteration_flops)

    return synflow_losses, flops_losses

def plot_results(experiments, result_type, legend_position='lower right'):
    for experiment in experiments:
        plt.plot(range(len(experiment)), experiment)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
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
    synflow_losses, flops_losses = parse_results(baseline_results)

# with open(log_path_attentive, 'r') as baseline_results:
#     avg_flops_attentive = parse_results(baseline_results)

# with open(log_path_attentive_synflow, 'r') as baseline_results:
#     avg_flops_attentive_synflow = parse_results(baseline_results)



plot_results(
    [smooth(synflow_losses, 0.9),], 
    'synflow_losses',
    'upper right')

plot_results(
    [smooth(flops_losses, 0.9),], 
    'flops_losses',
    'upper right')
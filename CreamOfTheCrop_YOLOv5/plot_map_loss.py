from pathlib import Path

import matplotlib.pyplot as plt

experiments_path = Path('experiments') / 'workspace' / 'train'
experiment_graphs = Path('experiment_results')

log_path_baseline = experiments_path / 'Baseline' / 'train.log'
log_path_attentive = experiments_path / 'MinLossAttentive_with_0.75' / 'train.log'
log_path_attentive_synflow = experiments_path / 'MaxSynflowFirstAttentive' / 'train.log'


def parse_results(results_file):
    mAPs, losses = [], []
    for line in results_file:
        if 'map' in line:
            mAPs.append(float(line.split('map')[1].split('Time')[0].split('(')[1].strip()[:-1]))
            # mAPs.append(float(line.split('map')[1].split('Time')[0].split('(')[0].split(' ')[2]))
        if 'Loss' in line:
            losses.append(float(line.split('Loss')[1].split('Time')[0].split('(')[0].split(':')[1].strip()))
    return mAPs, losses

def plot_results(experiments, result_type, legend_position='lower right'):
    ax = plt.gca()

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13)

    for experiment in experiments:
        plt.plot(range(len(experiment)), experiment)
    plt.xlabel('Mini-batches', fontdict={'fontsize': 15})
    plt.ylabel(f'SuperNet {result_type}', fontdict={'fontsize': 15})
    # plt.legend(['Uniform Sampling', 'Attentive Sampling(synflow)', 'Attentive Sampling(synflow)'], loc=legend_position)
    leg = ax.legend(['Uniform Sampling', 'Attentive Sampling(synflow)', 'Attentive Sampling(synflow)'], prop={"size":14}, loc=legend_position)
    plt.tight_layout()
    
    plt.savefig(experiment_graphs / f'{result_type}.pdf')
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
    mAPs, losses = parse_results(baseline_results)

with open(log_path_attentive, 'r') as attentive_results:
    attentive_mAPs, attentive_losses = parse_results(attentive_results)

with open(log_path_attentive_synflow) as attentive_synflow_results:
    attentive_synflow_mAPs, attentive_synflow_losses = parse_results(attentive_synflow_results)

# plot_results(smooth(mAPs[481:], 0.98), smooth(attentive_mAPs[148:], 0.98), 'validation mAP')
# plot_results(smooth(losses[481:], 0.98), smooth(attentive_losses[148:], 0.98), 'training Loss', legend_position='upper right')
plot_results(
    [smooth(mAPs, 0.98), smooth(attentive_synflow_mAPs, 0.98)], 
    'validation mAP', 
    legend_position='lower right'
)

plot_results(
    [smooth(losses[25:], 0.98), smooth(attentive_losses[25:], 0.98)], 
    'training Loss', 
    legend_position='upper right'
)





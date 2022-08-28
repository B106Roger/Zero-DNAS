from pathlib import Path

import matplotlib.pyplot as plt

experiments_path = Path('experiments') / 'workspace' / 'train'
experiment_graphs = Path('experiment_results')

log_path_baseline = experiments_path / 'WotAllFlops5.7Params30.1' / 'train.log'
log_path_attentive = experiments_path / 'Flops5.7Params30.1Meta' / 'train.log'
log_path_attentive_synflow = experiments_path / 'SynflowCacheDifferentBlocks' / 'train.log'


def parse_results(results_file):
    flops_loss, params_loss, losses, wot_loss, temperature = [], [], [], [], []
    for line in results_file:
        if 'Loss' in line:
            losses.append(float(line.split('Loss')[1].split(' ')[1]))

        if 'Flops_loss' in line:
            flops_loss.append(float(line.split('Flops_loss')[1].split('map')[0].split('(')[1].split(')')[0]))
            # mAPs.append(float(line.split('map')[1].split('Time')[0].split('(')[0].split(' ')[2]))
        if 'Overall_loss' in line:
            losses.append(float(line.split('Overall_loss')[1].split(':')[1].split('temperature')[0].strip()))
        
        if 'Params_loss' in line:
            params_loss.append(float(line.split('Params_loss')[1].split(' ')[2]))

        if 'Wot_loss' in line:
            wot_loss.append(float(line.split('Wot_loss')[1].split(' ')[1]))

        if 'temperature' in line:
            temperature.append(float(line.split('temperature')[1].split(' ')[2]))
            
    # return flops_loss, params_loss, synflow_loss, losses, temperature
    return flops_loss, params_loss, wot_loss, losses, []
def plot_results(experiments, result_type, legend_position='lower right'):
    for experiment in experiments:
        plt.plot(range(len(experiment)), experiment)
    
    plt.xlabel('Iterations')
    plt.ylabel(f'SuperNet {result_type}')
    plt.legend(['Baseline', 'Meta'], loc=legend_position)
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
    flops_loss, params_loss, wot_loss, losses, temperature = parse_results(baseline_results)

# with open(log_path_attentive, 'r') as attentive_results:
#    flops_loss_meta, params_loss_meta, synflow_loss_meta, losses_meta, temperature_meta = parse_results(attentive_results)

# with open(log_path_attentive_synflow) as attentive_synflow_results:
#     attentive_synflow_mAPs, attentive_synflow_losses = parse_results(attentive_synflow_results)

plot_results([smooth(flops_loss, 0.98)], 'WOT_Flops_lossFlops5.7Params30.1_epochs40_t3')
plot_results([smooth(wot_loss, 0.98)], 'WOT_Wot_lossFlops5.7Params30.1_epochs40_t3')
# # plot_results([smooth(temperature, 0.98)], 'Temperature_100epochs')
plot_results([smooth(params_loss, 0.98)], 'WOT_Params_lossFlops5.7Params30.1_epochs40_t3')
# plot_results([smooth(synflow_loss, 0.98)], 'Synflow_loss_constant_temp_50epoochs')

# plot_results([smooth(flops_loss, 0.98), smooth(flops_loss_meta, 0.98)], 'Flops_loss_Meta_compare_flops5.7_params30.1', legend_position='upper right')
# plot_results([smooth(losses, 0.98), smooth(losses_meta, 0.98)], 'Overall_loss_Meta_compare_flops5.7_params30.1', legend_position='upper right')
# plot_results([smooth(params_loss, 0.98), smooth(params_loss_meta, 0.98)], 'Params_loss_Meta_compare_flops5.7_params30.1', legend_position='upper right')
# plot_results([smooth(synflow_loss, 0.98), smooth(synflow_loss_meta, 0.98)], 'Synflow_loss_Meta_compare_flops5_params35', legend_position='upper right')
# plot_results(
#     [smooth(mAPs, 0.98), smooth(attentive_mAPs, 0.98), smooth(attentive_synflow_mAPs, 0.98)], 
#     'validation mAP', 
#     legend_position='upper left'
# )

# plot_results(
#     [smooth(losses[25:], 0.98), smooth(attentive_losses[25:], 0.98)], 
#     'training Loss', 
#     legend_position='upper right'
# )





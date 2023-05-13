import matplotlib.pyplot as plt
import numpy as np

# experiment_names = ['YOLO-DMaskingChannel-DNAS-25-V2-decay1', 'YOLO-DMaskingChannel-DNAS-25-V2-decay2', 'YOLO-DMaskingChannel-DNAS-25-V2-decay2-arch_no_gumbel', 'YOLO-DMaskingChannel-DNAS-25-V2-decay2-arch_no_gumbel-softmax-alignment']
# experiment_describe = ['decay_0.1','decay_1.0', 'decay_1.0_no_gumbel', 'decay_1.0_no_gumbel,temp_softmax']
# experiment_config = [False, False, True, True]
# experiment_root_dir = 'train'


# experiment_names = ['YOLO-DMaskingChannel-DNAS-25-Error', 'YOLO-DMaskingChannel-DNAS-35-Error', 'YOLO-DMaskingChannel-DNAS-45-Error']

# experiment_names = ['DMasking-25', 'DMasking-35', 'DMasking-45']
# experiment_describe = ['DNAS-25', 'DNAS-35', 'DNAS-45']
# experiment_config = [False, False, False]
# experiment_root_dir = 'Experiment-Project/DMaskingNAS'

experiment_names = ['DMaskingChannel-DNAS-25-LR-2-3-120-no-temp', 'DMaskingChannel-DNAS-35-LR-2-3-120-no-temp', 'DMaskingChannel-DNAS-45-LR-2-3-120-no-temp']
experiment_describe = ['25-GFLOPS', '35-GFLOPS','45-GFLOPS']
experiment_config = [False, False, False]
experiment_root_dir = 'train'
FREEZE_EPOCH=40



def get_experiment_mAPs(experiment_name):
    train_log_filename = f'experiments/workspace/{experiment_root_dir}/{experiment_name}/train.log'
    with open(train_log_filename) as f:
        train_log = f.readlines()
    mAPs = []

    for line in train_log:
        if 'all' in line:
            split_line = line.split()
            mAPs.append(float(split_line[-2]))

    return mAPs

plt.xlabel('epoch')
plt.ylabel('AP50')

for i, experiment_name in enumerate(experiment_names):
    mAPs = get_experiment_mAPs(experiment_name)
    offset = np.array(list(range(len(mAPs))))

    if experiment_config[i]:
        offset += FREEZE_EPOCH
    plt.plot(offset, mAPs)

plt.axvline(x=FREEZE_EPOCH, color='r', ls='--', lw=2, label='unfreeze architecture')
plt.tight_layout()
plt.legend(experiment_describe)

plt.savefig(f'fig.jpg')

# times = [54, 99, 1.8]
# scaling = 4
# experiments = [
#     {'name': 'CreamOfTheCrop', 'mAP': 79.5, 'FPS': 50, 'time_spent': 54}, 
#     {'name': 'AttentiveNAS', 'mAP': 80.6, 'FPS': 53, 'time_spent': 99}, 
#     {'name': 'Ours', 'mAP': 81.1, 'FPS': 58, 'time_spent': 1.8 * scaling},
# ]

# for experiment in experiments:
#     mAP = experiment['mAP']
#     fps = experiment['FPS']
#     time_spent = experiment['time_spent']
#     plt.scatter(x=fps, y=mAP, s=time_spent*2.4);

# plt.xlabel('FPS')
# plt.ylabel('mAP')

# plt.tight_layout()
# plt.legend([experiment['name'] for experiment in experiments])

# plt.savefig(f'experiment_results/model_comparison/{experiment_name}.png')
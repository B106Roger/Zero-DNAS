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

experiment_names = ['EA-SNIP-35-V2']
experiment_describe = ['SNIP-35', ]
experiment_config = [False]
experiment_root_dir = 'train'
FREEZE_EPOCH=40



def get_experiment_mAPs(experiment_name):
    train_log_filename = f'experiments/workspace/{experiment_root_dir}/{experiment_name}/train.log'
    with open(train_log_filename) as f:
        train_log = f.readlines()
    
    flops = []
    snips = []
    for line in train_log:
        if 'Current-Pool-FLOPS' in line:
            split_line = line.split()[5:]
            flops.append(np.array([float(item) for item in split_line]))
        if 'Current-Pool-SNIP' in line:
            split_line = line.split()[5:]
            snips.append(np.array([float(item) for item in split_line]))
    return flops, snips


plt.figure(figsize=(12, 9), dpi=80)

for i, experiment_name in enumerate(experiment_names):
    iter_flops, iter_snips = get_experiment_mAPs(experiment_name)
    iter_flops = np.array(iter_flops)
    iter_snips = np.array(iter_snips)
    iter_idx   = np.zeros_like(iter_flops)
    for ii in range(len(iter_idx)): iter_idx[ii] = ii*100
    
    iter_flops = iter_flops.flatten()
    iter_snips = iter_snips.flatten()
    iter_idx = iter_idx.flatten()
    
    mkder_list = []
    for l,h,c in [(0,25,'r'),(25,35,'g'),(35,45,'b')]:
        mask = np.logical_and(l<iter_flops, iter_flops<h)
        f = iter_flops[mask]
        s = iter_snips[mask]
        xi= iter_idx[mask]
        
        if len(f) == 0:continue
        ax=plt.plot(xi, s, c+'o')
        mkder_list.append(ax[0])
        
    plt.xlabel('AE iteration')
    plt.ylabel('SNIP')
    plt.title(f'{experiment_name} FLOP Distribution')

# plt.tight_layout()
# print(tuple(*mkder_list))
print(mkder_list)
plt.legend(
    (mkder_list[0], mkder_list[1], mkder_list[2]),
    ('FLOPS ~25','FLOPS 25~35','FLOPS 35~45')
    
)
# plt.legend(, color='r')
# plt.legend(, color='g')
# plt.legend(, color='b')


plt.savefig(f'fig.jpg')

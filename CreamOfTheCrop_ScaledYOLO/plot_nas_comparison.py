import matplotlib.pyplot as plt

experiment_name = 'different_nas_comparison'
times = [54, 99, 1.8]
scaling = 4
experiments = [
    {'name': 'CreamOfTheCrop', 'mAP': 79.5, 'FPS': 50, 'time_spent': 54}, 
    {'name': 'AttentiveNAS', 'mAP': 80.6, 'FPS': 53, 'time_spent': 99}, 
    {'name': 'Ours', 'mAP': 81.1, 'FPS': 58, 'time_spent': 1.8 * scaling},
]

for experiment in experiments:
    mAP = experiment['mAP']
    fps = experiment['FPS']
    time_spent = experiment['time_spent']
    plt.scatter(x=fps, y=mAP, s=time_spent*2.4);

plt.xlabel('FPS')
plt.ylabel('mAP')

plt.tight_layout()
plt.legend([experiment['name'] for experiment in experiments])

plt.savefig(f'experiment_results/model_comparison/{experiment_name}.png')
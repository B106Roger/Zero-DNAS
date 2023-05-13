import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 15
matplotlib.rc('font', size=15)
matplotlib.rc('axes', titlesize=15)

experiment_name = 'DNAS'
csp_names = ['YOLOv4', 'YOLOv4-P5']
v5_names = ['YOLOv5-M', 'YOLOv5-L']
nas_names = ['DNAS-25', 'DNAS-35', 'DNAS-45']

mAP_csp = [71.9, 72.5]
# mAP_v5 = [73.9, 75]
mAP_nas = [68.8, 70.9, 71.7]
# mAP_mobile = [68]
# mAP_ghost = [67]

flop_csp = [50.7, 70.3]
# flop_v5 = [99, 82]
flop_nas = [26.2, 35.5, 46.1]
# flop_mobile = [77]
# flop_ghost = [61]

# Parameter Millon
p_csp = [64.2, 87.1]
p_v5  = []
p_nas = [28.4, 38.4, 47.8]

offset_fps = 0
offset_mAP = 0.1
plt.plot(flop_csp, mAP_csp, marker='o')
for i, (fps, mAP) in enumerate(zip(flop_csp, mAP_csp)):

    if csp_names[i] == 'YOLOv4':
        offset_fps -= 3
        # offset_mAP -= 1.2
        pass
    if csp_names[i] == 'YOLOv4-P5':
        offset_fps -= 3
        # offset_mAP += 0.5
        pass
    plt.annotate(csp_names[i], (fps + offset_fps, mAP + offset_mAP))

# plt.plot(flop_v5, mAP_v5, marker='o')
# for i, (fps, mAP) in enumerate(zip(flop_v5, mAP_v5)):
#     offset_fps = 1
#     offset_mAP = 0.1

#     if v5_names[i] == 'YOLOv5-L':
#         offset_fps -= 5
#         offset_mAP -= 1.5

#     plt.annotate(v5_names[i],(fps + offset_fps, mAP + offset_mAP))

plt.plot(flop_nas, mAP_nas, marker='o')
for i, (fps, mAP) in enumerate(zip(flop_nas, mAP_nas)):
    plt.annotate(nas_names[i], (fps + offset_fps, mAP + offset_mAP), weight='bold')

# plt.plot(flop_mobile, mAP_mobile, marker='o')
# plt.annotate("YOLO-MobileNetv3",(flop_mobile[0] - 6, mAP_mobile[0] + 0.5))
# plt.plot(flop_ghost, mAP_ghost, marker='o')
# plt.annotate("YOLO-GhostNet",(flop_ghost[0] - 6, mAP_ghost[0] + 0.5))

plt.xlabel('GFLOP')
plt.ylabel('mAP')
plt.ylim(top=73)
plt.xlim(left=20, right=90)
plt.legend(['ScaledYOLO', 'YOLO-DNAS', 'Ours'])

plt.tight_layout()
plt.savefig(f'experiment_results/{experiment_name}2.png')
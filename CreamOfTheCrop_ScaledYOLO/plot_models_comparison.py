import matplotlib.pyplot as plt

experiment_name = 'csp_v5_nas_comparison'
csp_names = ['ScaledYOLOv4', 'YOLOv4-P5']
v5_names = ['YOLOv5-M', 'YOLOv5-L']
nas_names = ['NAS-SS', 'NAS-S', 'NAS-M', 'NAS', 'NAS-L']

mAP_csp = [80.8, 83.8]
mAP_v5 = [73.9, 75]
mAP_nas = [76, 78.5, 80, 81.1, 83.5]
mAP_mobile = [68]
mAP_ghost = [67]

fps_csp = [48, 35]
fps_v5 = [99, 82]
fps_nas = [83, 66, 61, 58, 46]
fps_mobile = [77]
fps_ghost = [61]

plt.plot(fps_csp, mAP_csp, marker='o')
for i, (fps, mAP) in enumerate(zip(fps_csp, mAP_csp)):
    offset_fps = 0
    offset_mAP = 0
    if csp_names[i] == 'ScaledYOLOv4':
        offset_fps -= 6
        offset_mAP -= 1.2

    if csp_names[i] == 'YOLOv4-P5':
        offset_fps -= 5
        offset_mAP += 0.5

    plt.annotate(csp_names[i], (fps + offset_fps, mAP + offset_mAP))

plt.plot(fps_v5, mAP_v5, marker='o')
for i, (fps, mAP) in enumerate(zip(fps_v5, mAP_v5)):
    offset_fps = 1
    offset_mAP = 0.1

    if v5_names[i] == 'YOLOv5-L':
        offset_fps -= 5
        offset_mAP -= 1.5

    plt.annotate(v5_names[i],(fps + offset_fps, mAP + offset_mAP))

plt.plot(fps_nas, mAP_nas, marker='o')
for i, (fps, mAP) in enumerate(zip(fps_nas, mAP_nas)):
    plt.annotate(nas_names[i], (fps + 1, mAP + 0.2), weight='bold')

plt.plot(fps_mobile, mAP_mobile, marker='o')
plt.annotate("YOLO-MobileNetv3",(fps_mobile[0] - 6, mAP_mobile[0] + 0.5))
plt.plot(fps_ghost, mAP_ghost, marker='o')
plt.annotate("YOLO-GhostNet",(fps_ghost[0] - 6, mAP_ghost[0] + 0.5))

plt.xlabel('FPS')
plt.ylabel('mAP')
plt.ylim(top=90)
plt.xlim(left=27, right=115)
plt.legend(['ScaledYOLO', 'YOLOv5', 'Ours'])

plt.tight_layout()
plt.savefig(f'experiment_results/model_comparison/{experiment_name}.png')
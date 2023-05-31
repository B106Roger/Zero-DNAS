# YOLOv4-large


## Installation

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3

# install mish-cuda, if you use different pytorch version, you could try https://github.com/thomasbrandon/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

## Testing

```
# download {yolov4-p5.pt, yolov4-p6.pt, yolov4-p7.pt} and put them in /yolo/weights/ folder.
python test.py --img 896 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p5.pt
python test.py --img 1280 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p6.pt
python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p7.pt
```

## Training

We use multiple GPUs for training.
{YOLOv4-P5, YOLOv4-P6, YOLOv4-P7} use input resolution {896, 1280, 1536} for training respectively.
```
# yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights '' --sync-bn --device 0,1,2,3 --name yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights 'runs/exp0_yolov4-p5/weights/last_298.pt' --sync-bn --device 0,1,2,3 --name yolov4-p5-tune --hyp 'data/hyp.finetune.yaml' --epochs 450 --resume
```

If your training process stucks, it due to bugs of the python.
Just `Ctrl+C` to stop training and resume training by:
```
# yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights 'runs/exp0_yolov4-p5/weights/last.pt' --sync-bn --device 0,1,2,3 --name yolov4-p5 --resume
```

## Citation

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>

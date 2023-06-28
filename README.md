# Zero-DNAS

## File Structure
----
```
Zero-DNAS
├── CreamOfTheCrop_ScaledYOLO
│   └── data
│       ├── obj-st  (Link)
│       └── VOC2007 (Link) 
├── ScaledYOLOv4
│   └── data
│       ├── coco    (Link)
│       ├── obj-st  (Link)
│       └── VOC2007 (Link) 
│
├── CreamOfTheCrop_YOLOv5   (under testing)
│   └── data
│       ├── obj-st  (Link)
│       └── VOC2007 (Link) 
└── yolov5                  (not available)
    ├── data (Link)
    └── venv (x)
```
# Detail
- Search
    - ZeroDNAS
    - DMaskingNAS
    - Zero-Cost EA
- Train
    - ZeroDNAS: 416x416
    - DNAS: 416x416
----
# Execution Step (For ScaledYOLOv4)
## 1 Search Command

```
cd CreamOfTheCrop_ScaledYOLO/
# Training YOLO-DMaskingNAS
python tools/train_dmasking.py --cfg config/search/train_dnas.yaml --data config/dataset/voc_dnas.yaml --hyp config/training/hyp.scratch.yaml --model config/model/Search-YOLOv4-P5.yaml --device GPU_ID --exp_name EXP_NAME --nas HARDWARE_CONSTRAINT

# Train Zero-Cost EA
python ./tools/train_zero_cost.py --cfg config/search/train_dnas.yaml --data config/dataset/voc_dnas.yaml --hyp config/training/hyp.scratch.yaml --model config/model/Search-YOLOv4-P5.yaml --device GPU_ID --exp_name EXP_NAME  --nas HARDWARE_CONSTRAINT --zc snip

# Train ZeroDNAS
python tools/train_zdnas.py --cfg config/search/train_zdnas.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model ./config/model/Search-YOLOv4-CSP.yaml --device GPU_ID --exp_name EXP_NAME --nas HARDWARE_CONSTRAINT -zc wot

# Train Zero-DNAS
TO-BE-CONTINUE

```

- data: change the data.yaml file according the dataset you want to search for. (voc.yaml, coco.yaml ......)
- exp_name: name your experiment, later some of information during training would be store in the `CreamOfTheCrop_ScaledYOLO/experiments/workspace/train/{exp_name}`
    - All Training Algorihtm
        - alpha_distirbution.txt: the distribution of architecture parameter for each epoch.
        - beta_distribution.txt: the distribtuion of architecture parameter (normalized by softmax) for each epochs.
        - train.log: the training loss for a period of training steps.
        - model: derived model architecture


## 2 Train the model
python train.py --batch-size 16 --img 416 416 --data ./data/voc.yaml --cfg YOUR_MODIFIED_MODEL.yaml --weights '' --name VOC-NAS-L --hyp ./data/hyp.finetune.yaml

- cfg: model config. should be something like (yolov4-p5.yaml, yolov4-csp.yaml, yolov4-csp-search.yaml)

----


## 3 Extra Helper Function
### 2-1 Plot Theta Distribtuion
```
python ./plot_thetas.py --exp_name VOC-NAS-SS --sp small
```
- exp_name: the experiment name, should same as the previous argument exp_name in tools/train.py 
- sp: describe the your search space
    - for small search space: depth is provided with [0,6,4,2] options
    - for large search sapce: depth is provided with [8,6,4,2] options
- After run the program, it would generate ```./experiments/workspace/train/{exp_name}/thetas.png```

### 2-2 (Optional) Plot Theta Distribution during training
- generate a GIF file for the change of beta-distribtion during training.
```
python ./plot_thetas_gif.py --exp_name VOC-NAS-SS --sp small
```
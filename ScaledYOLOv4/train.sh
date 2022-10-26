python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 32 --img 416 416 \
 --data ./data/voc.yaml --cfg ./models/MODEL_CONFIGURATION \
 --weights ''  --device 1,2 --name hardware-aware-wot-f5-p32

export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/nvvm:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda/lib64:/usr/local/cuda/nvvm:$PATH"
python -m torch.distributed.launch --nproc_per_node 2 --master_port 47770 train.py --batch-size 32 --img 416 416 --data ./data/voc.yaml --cfg ./models/VOC-NAS-SS-01.yaml --weights '' --device 1,3 --name VOC-NAS-SS-0
python train.py --batch-size 16 --img 416 416 --data ./data/voc.yaml --cfg ./models/VOC-NAS-L.yaml --weights '' --name VOC-NAS-L --hyp ./data/hyp.finetune.yaml
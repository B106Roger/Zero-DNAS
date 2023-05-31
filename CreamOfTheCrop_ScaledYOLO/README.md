python ./tools/train_zero_cost.py --cfg config/search/train_dnas.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model config/model/Search-YOLOv4-P5.yaml --device 2 --exp_name test  --nas DNAS-50 --zc snip

python ./tools/train_zero_cost.py --cfg config/search/train_dnas.yaml --data config/dataset/voc_dnas.yaml --hyp config/training/hyp.scratch.yaml --model config/model/Search-YOLOv4-CSP.yaml --device 4 --exp_name 0530-naswot-50  --nas DNAS-50 --zc naswot

python ./tools/train_dmasking.py --cfg config/search/train_dnas.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model config/model/Search-YOLOv4-P5.yaml --device 2 --exp_name test  --nas DNAS-50
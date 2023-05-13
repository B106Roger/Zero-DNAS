1. train.py => original zero-dans
    - search space = 12^4 * 4 ^ 4
    - train config = train.yaml
        - BottleneckCSP:  0
        - BottleneckCSP2: 0
        - SEARCH_RESOLUTION: 288 (原本的實驗有誤，使用288來計算supernet的flops數，而且並不是用真的flop，而是用mac[flop=mac*2])
2. train_dnas.py => YOLO-DNAS (FBNet and ScaledYOLOv4)
    - search space = 12^8
    - train config = train_dnas.yaml
        - BottleneckCSP:  0
        - BottleneckCSP2: 1
        - SEARCH_RESOLUTION: 416

3. train.py => original zero-dans (but search space increase)
    - search space => 12^8
    - train config = **train_zdnasV2.yaml**
        - BottleneckCSP:  0
        - BottleneckCSP2: 1
        - SEARCH_RESOLUTION: 288
4. train.py => original zero-dans
    - search space => 12^4 * 4 ^ 4
    - train config = **train_zdnas.yaml**
        - BottleneckCSP:  0
        - BottleneckCSP2: 0
        - SEARCH_RESOLUTION: 288

5. train_sensitive.py => my experiment with foreground and background sensitivity
    - search space => 12^8

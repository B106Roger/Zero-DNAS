import random
import os
import numpy as np

sep_path = '../data/coco/train2017.txt'
f=open(sep_path, 'r')
lines = f.readlines()
np.random.shuffle(lines)

sep_idx = int(len(lines) * 0.2)

train_weight_images = lines[sep_idx:]
train_thetas_images = lines[:sep_idx]


weight_file = open('./train_weight.txt', 'w')
weight_file.writelines(train_weight_images)
weight_file.close()

thetas_file = open('./train_thetas.txt', 'w')
thetas_file.writelines(train_thetas_images)
thetas_file.close()
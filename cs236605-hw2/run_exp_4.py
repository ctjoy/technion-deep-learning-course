# -*- coding: utf-8 -*-
import os
import re
import sys
import glob
import numpy as np
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.experiments as experiments
import hw2.models as models


seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed)
data_dir = os.path.join(os.getenv('HOME'), '.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

x0,_ = ds_train[0]
in_size = x0.shape
num_classes = 10
print('input image size =', in_size)

params = [
    {
        'filename': 'exp2_L1_K64-128-256-512',
        'K': [64,128,256,512],
        'L': 1
    },
    {
        'filename':'exp2_L2_K64-128-256-512',
        'K': [64,128,256,512],
        'L': 2
    },
    {
	'filename': 'exp2_L3_K64-128-256-512',
	'K': [64,128,256,512],
        'L': 3
    },
    {
        'filename': 'exp2_L4_K64-128-256-512',
        'K': [64,128,256,512],
        'L': 4
    },
    
]

for p in params:
    experiments.run_experiment(p['filename'],
                               seed=seed,
                               epochs=10,
                               early_stopping=5,
                               filters_per_layer=[p['K']],
                               layers_per_block=p['L'])

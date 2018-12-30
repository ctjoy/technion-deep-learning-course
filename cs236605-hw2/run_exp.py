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

file_name = 'exp1_1_K32_L2'
K = 32
L = 2
experiments.run_experiment(file_name,
                           seed=seed,
                           bs_train=50,
                           batches=10,
                           epochs=10,
                           early_stopping=5,
                           filters_per_layer=[K],
                           layers_per_block=L)

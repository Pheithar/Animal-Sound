import sys

# setting path
sys.path.append('../../')

from models import utils
from models import SimpleCNN

import matplotlib.pyplot as plt

import torch
import torchsummary
from torchvision import transforms
import torchaudio


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torch.utils.data import DataLoader

from sklearn.preprocessing import OneHotEncoder


import numpy as np
from importlib import reload

import yaml


# Parameters
random_seed = 42

batch_size = 16
lr = 0.001
epochs = 150

net_name = "FlattenCNN"

# Load YAML
with open("FlattenCNN.yaml", "r") as f:
  net_archi = yaml.load(f, Loader=yaml.FullLoader)


# Transformations for data augmentation and proper processing
transform_img = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
     torchaudio.transforms.FrequencyMasking(freq_mask_param=40),
     torchaudio.transforms.TimeMasking(time_mask_param=40),
    ]
)

# Load dataset
dataset = utils.SpectrogramDataset("../../datasets/full_dataset_df.csv",
                                   "../../spectrograms/full_dataset",
                                   "gbifID", "family",
                                   transform=transform_img,
                                   one_hot_encode_labels=True)
dataset.show_sample(8, figsize=(20, 10))

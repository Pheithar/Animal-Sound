import sys
sys.path.append('../')


from  models import utils, Simple_CNN

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
epochs = 2

net_name = "FlattenCNN"


with open("./yaml/FlattenCNN.yaml", "r") as f:
  net_archi = yaml.load(f, Loader=yaml.FullLoader)

# Dataset transformations
transform_img = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
     torchaudio.transforms.FrequencyMasking(freq_mask_param=40),
     torchaudio.transforms.TimeMasking(time_mask_param=40),
    ]
)

# Ctreate dataset
dataset = utils.SpectrogramDataset("../datasets/full_dataset_df.csv",
                                   "../spectrograms/full_dataset",
                                   "gbifID", "family",
                                   transform=transform_img,
                                   one_hot_encode_labels=True)
# dataset.show_sample(8, figsize=(20, 10))

# Create test and train dataset
test_percentage = .2
test_len = int((len(dataset) * test_percentage))
train_len = len(dataset) - test_len

train, test = random_split(dataset, [train_len, test_len], torch.Generator().manual_seed(random_seed))

train_loader = DataLoader(train, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)


# Create network, loss function and optimizer

net = Simple_CNN.SimpleCNN(net_archi[net_name])

net = utils.cuda_network(net)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = lr)


# Train the network
((train_loss, train_acc),
 (val_loss, val_acc)) = net.fit(epochs, train_loader, test_loader,
                                criterion, optimizer,
                                log_file="logs/" + net_name+".log",
                                plot_file="images/" + net_name+".png",
                                train_name=net_name, show=False)

# Print outputs

print("Train set:")
print(f"\tLoss: \t\t{train_loss:.2f}")
print(f"\tAccuracy: \t{train_acc:.2f}")

print("Validation set:")
print(f"\tLoss: \t\t{val_loss:.2f}")
print(f"\tAccuracy: \t{val_acc:.2f}")

# Save the network
torch.save(net, "networks/" + net_name+".pth")

import sys
sys.path.append('../')

from models import utils, Simple_CNN, Simple_LSTM, Hierarchical

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

import numpy as np

import yaml

import time

with open("./yamls/HierarchyMix.yaml", "r") as f:
  net_params = yaml.load(f, Loader=yaml.FullLoader)

net_organization = net_params["net_organization"]
node_params = net_params["node_params"]
net_architecture = net_params["net_architecture"]

random_seed = 42

net_name = "test"

transform_img = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]
)

transform_labels = transforms.Compose(
    [transforms.ToTensor()]
)

dataset = utils.HierarchicalDataset("../datasets/full_dataset_df.csv",
                                    "../spectrograms/full_dataset",
                                    "../mfcc/",
                                    "gbifID",
                                    ["phylum", "class", "family"],
                                    transform=transform_img, preload=True)

H = Hierarchical.HierarchicalClassification(net_architecture, node_params, net_organization, dataset)

start_time = time.time()

train_acc, test_acc = H.fit()

end_time = time.time()

total_time = end_time - start_time
hours = total_time // 3600
minutes = total_time % 3600 // 60
seconds = total_time % 60

print("Hierarchical Network")
print("Total training time:")
print(f"\t{hours:4.0f} hours, {minutes:4.0f} minutes and {seconds:4.2f} seconds")
print()
print("Train set:")
print(f"\tAccuracy: \t{train_acc:.2f}")

print("Test set:")
print(f"\tAccuracy: \t{test_acc:.2f}")

H.save()

with open("./logs/hierarchical/HierarchicalMix.txt", "w") as f:
    f.write("train accuracy, test accuracy\n")
    f.write(f"{train_acc}, {test_acc}")

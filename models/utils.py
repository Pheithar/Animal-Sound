import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from skimage import io, transform


import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir: dir of images
        path_column: 
    """
    def __init__(self, csv_path: str, root_dir: str, id_column: str, label_column: str, transform=None):
        self.csv = pd.read_csv(csv_path)

        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.col_label_idx = self.csv.columns.get_loc(label_column)

        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
            str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
        image = io.imread(img_name)
        
        label = self.csv.iloc[idx, self.col_label_idx]

        sample = {"image": image, "label": label}
        return sample
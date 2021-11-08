import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from skimage import io, transform

from sklearn.preprocessing import MultiLabelBinarizer


import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir: dir of images
        path_column: 
    """
    def __init__(self, csv_path: str, root_dir: str, id_column: str, label_column: str, transform=None, one_hot_encode_labels: bool = False):
        self.csv = pd.read_csv(csv_path)
        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.col_label_idx = self.csv.columns.get_loc(label_column)
        self.root_dir = root_dir

        self.transform = transform
        
        self.one_hot_encode_bool = one_hot_encode_labels
        
        # Giving a class id for each unique class in dataset
        self.lexicon = {class_name: class_id for class_id, class_name in enumerate(self.csv["class"].unique())}

        if one_hot_encode_labels:
            # Assigning label to each row
            labels = list(map(self.lexicon.get, self.csv[label_column]))
            self.labels = np.eye(len(self.lexicon), dtype="float")[labels]
        
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
            str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
        image = io.imread(img_name)


        if self.one_hot_encode_bool:
            label = self.labels[idx]
        else:
            label = self.lexicon[self.csv.iloc[idx, self.col_label_idx]]

        if self.transform: image = self.transform(image)
        if type(label) != torch.Tensor:
            label = torch.from_numpy(label)

        sample = {"image": image, "label": label}


        return sample
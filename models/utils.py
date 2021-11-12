import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from skimage import io, transform

from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt


### SPECTROGRAM DATASET CLASS ###
class SpectrogramDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir: dir of images
        id_column: column of the id
        label_column: column of the label
        transform: transformations of the image
        one_hot_encode_labels: whether or not to use one hot encoding
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

    def __getitem__(self, idx: int):
        """Return image and label"""

        img_name = os.path.join(self.root_dir,
            str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
        image = io.imread(img_name)


        if self.one_hot_encode_bool:
            label = self.labels[idx]
        else:
            label = [self.lexicon[self.csv.iloc[idx, self.col_label_idx]]]

        if self.transform: image = self.transform(image)

        sample = {"image": image, "label": np.float32(label)}


        return sample

    def get_image(self, idx: int):
        """Return image without transformations"""
        img_name = os.path.join(self.root_dir,
            str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
        image = io.imread(img_name)

        return image

    def get_label(self, idx: int):
        """Return original label"""
        return self.csv.iloc[idx, self.col_label_idx]


    def show_sample(self, sample_size: int = 4, show: bool = True,
                    file_name: str = None, samples_per_row: int = 2,
                    figsize: tuple = (20, 5)):
        """Display sample of the dataset"""
        sample_idx = np.random.choice(range(len(self)), sample_size)

        num_rows = int(np.ceil(sample_size / samples_per_row))

        fig, axarr = plt.subplots(num_rows, samples_per_row, figsize=figsize)

        for idx, ax in zip(sample_idx, axarr.flatten()):
            image = self.get_image(idx)
            label = self.get_label(idx)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(label)

        plt.axis("off")

        if file_name:
            plt.savefig(file_name)
        if show:
            plt.show()
        else:
            plt.close()



### CUDA FUNCTIONS ###

def cuda_network(network):
    """Converts network to CUDA if available"""
    if torch.cuda.is_available():
        print('CUDA available: converting network to CUDA')
        network.cuda()
    return network

def get_cuda(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()

    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()

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
        self.lexicon = {class_name: class_id for class_id, class_name in enumerate(self.csv[label_column].unique())}

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
            image, _ = self[idx].values()

            # Permuting to put channels at the end
            image = image.squeeze().permute(1,2,0)

            # Normalizing between 0 and 1
            image = NormalizeData(image.numpy())
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Label={self.get_label(idx)}, idx: {idx}")

        plt.axis("off")

        if file_name:
            plt.savefig(file_name)
        if show:
            plt.show()
        else:
            plt.close()

### DATASET FIT FOR HIERARCHICAL STRUCTURE ###
class HierarchicalDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir: dir of images
        id_column: column of the id
        hierarchy: order of columns in hierarchy
        transform: transformations of the image
    """

    def __init__(self, csv_path: str, root_dir: str, id_column: str, hierarchy: list, transform = None):
        self.csv = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.transform = transform
        self.hierarchy = hierarchy

        self.order = []

        self.build_hierarchy_labels()


    def __len__(self):
        return len(self.csv)

    # Build a dictionary with the labels for each possible value in dataset
    def build_hierarchy_labels(self):

        self.labels = {}

        # Go from 0 to the number of hierarchy levels
        for i in range(len(self.hierarchy)):
            # Get the different values of level, keeping previous info
            for _, row in self.csv[self.hierarchy[:i+1]].drop_duplicates().iterrows():
                # Update the dataframe so ot only has the info of the proper
                # branch in the tree
                df = self.csv
                for level in self.hierarchy[:i]:
                    df = df[df[level] == row[level]]

                # Get unique elements and save the label
                unique_elements = df[self.hierarchy[i]].unique()

                # Save the labels the level of the row with its value
                label = np.array([0.0
                                 if x!=row[self.hierarchy[i]]
                                 else 1.0
                                 for x in unique_elements])

                self.labels[row[self.hierarchy[i]]] = label

    def set_order(self, order):
        self.order = order

    def get_label(self, idx: int):

        row = self.csv.iloc[idx]

        df = self.csv

        in_hierarchy = True

        df = self.csv[self.hierarchy].drop_duplicates()

        # Get only the distinct elements of 'order'
        for column, value in zip(self.hierarchy, self.order):
            df = df[df[column] == value]

        to_classify = self.hierarchy[len(self.order)]

        if row[to_classify] in df[to_classify].values:
            return self.labels[row[to_classify]]

        return np.zeros(len(self.labels[df[to_classify].values[0]]))


    def __getitem__(self, idx: int):
        img_name = os.path.join(self.root_dir,
            str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.get_label(idx)

        sample = {"image": image, "label": np.float32(label)}

        return sample


### NUMPY FUNCTIONS ###

# Normalizes data between 0 and 1
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

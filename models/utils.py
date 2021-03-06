import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F

from skimage import io, transform

from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt

import h5py

from tqdm import tqdm


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
    def __init__(self, csv_path: str, root_dir: str, id_column: str,
                label_column: str, transform=None,
                one_hot_encode_labels: bool = False,
                preload: bool = False):
        self.csv = pd.read_csv(csv_path)
        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.col_label_idx = self.csv.columns.get_loc(label_column)
        self.root_dir = root_dir
        self.preload = preload


        self.transform = transform

        if preload:
            tqdm.pandas()
            print("Loading into memory")
            self.csv["data"] = self.csv[id_column].progress_apply(lambda x: io.imread(os.path.join(root_dir, str(x) + ".jpg")))

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


        image = None
        if self.preload:
            image = self.csv.iloc[idx]["data"]
        else:
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


### MFCC DATASET CLASS ###
class MFCCDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir: dir of h5 file
        id_column: column of the id
        label_column: column of the label
        one_hot_encode_labels: whether or not to use one hot encoding
    """
    def __init__(self, csv_path: str, root_dir: str, id_column: str,
                label_column: str, one_hot_encode_labels: bool = False,
                preload: bool = False):
        self.csv = pd.read_csv(csv_path)
        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.col_label_idx = self.csv.columns.get_loc(label_column)
        self.root_dir = root_dir
        self.max_len = 0
        self.preload = preload

        for i, row in self.csv.iterrows():
            path = os.path.join(self.root_dir, str(row[id_column]) + ".npz")
            row_shape = np.load(path)["data"].shape[1]
            if row_shape > self.max_len:
                self.max_len = row_shape


        if preload:
            tqdm.pandas()
            print("Loading into memory")
            self.csv["data"] = self.csv[id_column].progress_apply(lambda x: np.load(os.path.join(root_dir, str(x) + ".npz"))["data"])

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

        mfcc = None
        if self.preload:
            mfcc = self.csv.iloc[idx]["data"]
        else:
            key = self.csv.iloc[idx, self.col_path_idx]

            path = os.path.join(self.root_dir, str(key) + ".npz")
            mfcc = np.load(path)["data"]

        mfcc = torch.tensor(mfcc)
        mfcc = F.pad(mfcc, (0, self.max_len-mfcc.shape[1]))

        if self.one_hot_encode_bool:
            label = self.labels[idx]
        else:
            label = [self.lexicon[self.csv.iloc[idx, self.col_label_idx]]]

        sample = {"mfcc": mfcc, "label": np.float32(label)}

        return sample


### DATASET FIT FOR HIERARCHICAL STRUCTURE ###
class HierarchicalDataset(Dataset):
    """
    Args:
        csv_path: path to csv file
        root_dir_img: dir of images
        root_dir_np: dir of mfcc
        id_column: column of the id
        hierarchy: order of columns in hierarchy
        transform: transformations of the image
    """

    def __init__(self, csv_path: str, root_dir_img: str, root_dir_np: str,
                id_column: str, hierarchy: list, transform = None,
                preload: bool = False):
        self.csv = pd.read_csv(csv_path)
        self.root_dir_img = root_dir_img
        self.root_dir_np = root_dir_np
        self.col_path_idx = self.csv.columns.get_loc(id_column)
        self.transform = transform
        self.hierarchy = hierarchy
        self.max_len = 0
        self.preload = preload
        self.order_hierarchy = {}

        for i, row in self.csv.iterrows():
            path = os.path.join(self.root_dir_np, str(row[id_column]) + ".npz")
            row_shape = np.load(path)["data"].shape[1]
            if row_shape > self.max_len:
                self.max_len = row_shape


        if preload:
            tqdm.pandas()
            print("Loading images into memory")
            self.csv["data_img"] = self.csv[id_column].progress_apply(lambda x: io.imread(os.path.join(root_dir_img, str(x) + ".jpg")))
            print("Loading mfcc into memory")
            self.csv["data_mfcc"] = self.csv[id_column].progress_apply(lambda x: np.load(os.path.join(root_dir_np, str(x) + ".npz"))["data"])

        self.order = []

        self.build_hierarchy_labels()

        self.flatten_last_level()

    def __len__(self):
        return len(self.csv)

    def flatten_last_level(self):
        last_level = self.hierarchy[-1]
        self.flatten_labels = {}

        flatten_last_level = self.csv[last_level].drop_duplicates().values

        for i, leaf in enumerate(flatten_last_level):
            self.flatten_labels[leaf] = np.zeros(flatten_last_level.shape)
            self.flatten_labels[leaf][i] = 1


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
                if i > 0:
                    key = row[self.hierarchy[i-1]]
                    if self.order_hierarchy.get(key):
                        self.order_hierarchy[key] += [row[self.hierarchy[i]]]
                    else:
                        self.order_hierarchy[key] = [row[self.hierarchy[i]]]


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

        mfcc = None
        image = None
        if self.preload:
            image = self.csv.iloc[idx]["data_img"]
            mfcc = self.csv.iloc[idx]["data_mfcc"]
        else:
            img_name = os.path.join(self.root_dir_img,
                str(self.csv.iloc[idx, self.col_path_idx]) + ".jpg")
            image = io.imread(img_name)

            key = self.csv.iloc[idx, self.col_path_idx]

            path = os.path.join(self.root_dir_np, str(key) + ".npz")
            mfcc = np.load(path)["data"]

        mfcc = torch.tensor(mfcc)
        mfcc = F.pad(mfcc, (0, self.max_len-mfcc.shape[1]))

        if self.transform:
            image = self.transform(image)

        label = self.get_label(idx)

        last_level = self.csv.iloc[idx][self.hierarchy[-1]]
        leaf_label = self.flatten_labels[last_level]

        sample = {
                    "image": image,
                    "mfcc": mfcc,
                    "label": np.float32(label),
                    "leaf_label": leaf_label
                }

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

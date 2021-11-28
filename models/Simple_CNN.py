import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score

from . import utils

import warnings
warnings.filterwarnings("ignore")


class SimpleCNN(nn.Module):
    # Init
    def __init__(self, net_arch: dict):

        super(SimpleCNN, self).__init__()
        # Fields
        input_channels, height, width = net_arch["input_shape"]

        # Check format so for loops do not do it wrong with 'zip'
        assert len(net_arch["conv_channels"]) == len(net_arch["conv_kernel_size"]),\
        f"Lenght of convolutional channels ({len(net_arch['conv_channels'])})"\
        + f" must be the same as the lenght of convolutional kernel size"\
        f" ({len(net_arch['conv_kernel_size'])})."

        assert len(net_arch["conv_channels"]) == len(net_arch["conv_dropout"]),\
        f"Lenght of convolutional channels ({len(net_arch['conv_channels'])})"\
        f" must be the same as the lenght of convolutional dropout"\
        f" ({len(net_arch['conv_dropout'])})."

        assert len(net_arch["conv_channels"]) == len(net_arch["pooling_size"]),\
        f"Lenght of convolutional channels ({len(net_arch['conv_channels'])})"\
        f" must be the same as the lenght of convolutional dropout"\
        f" ({len(net_arch['pooling_size'])})."

        assert len(net_arch["linear_features"]) == 1 + len(net_arch["linear_dropout"]),\
        f"Lenght of linear features ({len(net_arch['linear_features'])})"\
        f" must be one more than the the lenght of linear dropout"\
        f" ({len(net_arch['linear_dropout'])})."


        # Layers - Conv
        self.conv = nn.ModuleList()
        self.bnorm = nn.ModuleList() # Batch Normalization
        self.cdrop = nn.ModuleList() # Dropout
        self.pool = nn.ModuleList()

        for channels, kernel_size, drop, pool in zip(net_arch["conv_channels"],
                                                     net_arch["conv_kernel_size"],
                                                     net_arch["conv_dropout"],
                                                     net_arch["pooling_size"]):

            layer = nn.Conv2d(in_channels=input_channels, out_channels=channels,
                              kernel_size=kernel_size, padding="same")
            self.conv.append(layer)
            self.bnorm.append(nn.BatchNorm2d(channels))
            self.cdrop.append(nn.Dropout(drop))
            self.pool.append(nn.MaxPool2d(pool))

            # Update parameters
            input_channels = channels
            height = int(height / pool[0])
            width = int(width / pool[1])

        # Layers - Linear
        self.dense = nn.ModuleList()
        self.ldrop = nn.ModuleList()

        # Flatten
        input_channels = input_channels * height * width


        for linear_features in net_arch["linear_features"]:
            layer = nn.Linear(input_channels, linear_features)
            self.dense.append(layer)

            # Update parameters
            input_channels = linear_features

        for drop in net_arch["linear_dropout"]:
            self.ldrop.append(nn.Dropout(drop))

        # Activation
        self.relu = nn.ReLU()
        self.last_activation = eval("nn." + net_arch["last_layer_activation"])

        # Flatten
        self.flatten = nn.Flatten()

    def forward(self, x):

        for conv, pool, batch_norm, dropout in zip(self.conv, self.pool,
                                                   self.bnorm, self.cdrop):
            x = conv(x)
            x = batch_norm(x)
            x = pool(x)
            x = self.relu(x)
            x = dropout(x)

        x = self.flatten(x)

        for dense, dropout in zip(self.dense, self.ldrop):
            x = dense(x)
            x = self.relu(x)
            x = dropout(x)

        x = self.last_activation(self.dense[-1](x))

        return x

    def fit(self, num_epochs: int, train_loader: DataLoader,
                   validation_loader: DataLoader,
                   criterion, optimizer,
                   show: bool = True, frequency_val : int = 2,
                   log_file: str = None, plot_file: str = None,
                   train_name: str = "Network"):

        sns.set()

        if log_file:
            with open(log_file, "a") as f:
                f.write("-------------------------------------------\n")
                f.write(train_name + "\n")

        train_loss = []
        train_acc = []
        plot_epochs_train = []

        val_loss = []
        val_acc = []
        plot_epochs_val = []

        fig, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(12, 10))

        for epoch in tqdm(range(num_epochs), desc=f"Training {train_name}"):

            running_loss = 0.0
            running_accuracy = 0.0
            # State that the network is in train mode
            self.train()
            for i, data in enumerate(train_loader):

                # Get the inputs
                inputs, labels = data.values()

                inputs = utils.get_cuda(inputs)
                labels = utils.get_cuda(labels)

                # Make them variables
                optimizer.zero_grad()

                # Forward, backward and around
                outputs = self(inputs)

                # print(f"out: {outputs.shape}, lab: {labels.shape}")

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                labels = utils.get_numpy(self.compute_prediction(labels))
                outputs = utils.get_numpy(self.compute_prediction(outputs))


                running_accuracy += accuracy_score(outputs, labels)
                running_loss += loss.data.item()


            train_loss.append(running_loss / len(train_loader))
            train_acc.append(running_accuracy / len(train_loader))
            plot_epochs_train.append(epoch+1)


            if epoch % frequency_val == 0 or epoch == num_epochs-1:
                running_loss = 0.0
                running_accuracy = 0.0

                # State that the network is in validation mode
                self.eval()
                for i, data in enumerate(validation_loader):

                    # Get the inputs
                    inputs, labels = data.values()

                    inputs = utils.get_cuda(inputs)
                    labels = utils.get_cuda(labels)

                    outputs = self(inputs)

                    loss = criterion(outputs, labels)

                    labels = utils.get_numpy(self.compute_prediction(labels))
                    outputs = utils.get_numpy(self.compute_prediction(outputs))

                    running_accuracy += accuracy_score(outputs, labels)
                    running_loss += loss.data.item()

                val_loss.append(running_loss / len(validation_loader))
                val_acc.append(running_accuracy / len(validation_loader))
                plot_epochs_val.append(epoch+1)

                if log_file:
                    with open(log_file, "a") as f:
                        f.write("-------------------------------------------\n")
                        f.write(f"Epoch {epoch+1}:\n")
                        f.write(f"\t Train Loss: {train_loss[-1]:.2f}\n")
                        f.write(f"\t Train Accuracy: {train_acc[-1]:.2f}\n")
                        f.write(f"\t Validation Loss: {val_loss[-1]:.2f}\n")
                        f.write(f"\t Validation Accuracy: {val_acc[-1]:.2f}\n")

        step = max(int(len(plot_epochs_train) // 10), 1)

        loss_ax.set_title("Loss function value in the train and validation sets")
        loss_ax.plot(plot_epochs_train, train_loss, label="Train Loss")
        loss_ax.plot(plot_epochs_val, val_loss, label="Validation Loss")
        loss_ax.set_xlabel("Epochs")
        loss_ax.set_ylabel("Value")
        loss_ax.legend()
        loss_ax.set_xticks(range(1, len(plot_epochs_train), step))

        acc_ax.set_title("Accuracy of the train and validation sets")
        acc_ax.plot(plot_epochs_train, train_acc, label="Train Accuracy")
        acc_ax.plot(plot_epochs_val, val_acc, label="Validation Accuracy")
        acc_ax.set_xlabel("Epochs")
        acc_ax.set_ylabel("Percentage")
        acc_ax.set_xticks(range(1, len(plot_epochs_train), step))
        acc_ax.legend()




        if plot_file:
            plt.savefig(plot_file)

        if show:
            plt.show()
        else:
            plt.close()

        return ((train_loss[-1], train_acc[-1]), (val_loss[-1], val_acc[-1]))


    def compute_prediction(self, y):
        # if y.shape[1] == 1:
        #     return y.int()

        return torch.max(y, 1)[1]

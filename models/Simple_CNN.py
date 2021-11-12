import torch
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import utils


class SimpleCNN(nn.Module):
    # Init
    def __init__(self, input_channels: int, input_shape: tuple,
                 num_labels: int, net_arch: dict):

        super(SimpleCNN, self).__init__()
        # Fields
        self.input_channels = input_channels
        self.height, self.width = input_shape
        self.num_labels = num_labels

        # Layers - Conv
        self.conv = nn.ModuleList()

        for channels, kernel_size in zip(net_arch["conv_channels"],
                                        net_arch["conv_kernel_size"]):
            layer = nn.Conv2d(in_channels=input_channels, out_channels=channels,
                              kernel_size=kernel_size, padding="same")
            self.conv.append(layer)

            # Update parameters
            input_channels = channels

        # Pooling
        self.pool = nn.ModuleList()

        for pool_size in net_arch["pooling_size"]:
            self.pool.append(nn.MaxPool2d(pool_size))

            # Update parameters
            self.height = int(self.height / pool_size[0])
            self.width = int(self.width / pool_size[1])

        # Layers - Linear
        self.dense = nn.ModuleList()

        input_channels = input_channels * self.height * self.width

        for linear_features in net_arch["linear_features"]:
            layer = nn.Linear(input_channels, linear_features)
            self.dense.append(layer)

            # Update parameters
            input_channels = linear_features

        self.dense.append(nn.Linear(input_channels, num_labels))


        # Activation
        self.relu = nn.ReLU()
        self.last_activation = eval("nn." + net_arch["last_layer_activation"])

        # Flatten
        self.flatten = nn.Flatten()

    def forward(self, x):

        for conv, pool in zip(self.conv, self.pool):
            x = self.relu(pool(conv(x)))

        x = self.flatten(x)

        for dense in self.dense[:-1]:
            x = self.relu(dense(x))

        x = self.last_activation(self.dense[-1](x))

        return x

    def fit(self, num_epochs: int, train_loader: DataLoader,
                   validation_loader: DataLoader,
                   criterion, optimizer,
                   show: bool = True, frequency_val : int = 2,
                   log_file: str = None, plot_file: str = None):
        train_loss = []
        train_acc = []
        plot_epochs_train = []

        val_loss = []
        val_acc = []
        plot_epochs_val = []

        fig, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(12, 10))

        for epoch in tqdm(range(num_epochs)):

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

                # print(outputs.shape, labels.shape)

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


        loss_ax.set_title("Loss function value in the train and validation sets")
        loss_ax.plot(plot_epochs_train, train_loss, label="Train Loss")
        loss_ax.plot(plot_epochs_val, val_loss, label="Validation Loss")
        loss_ax.set_xlabel("Epochs")
        loss_ax.set_ylabel("Value")
        loss_ax.set_xticks(plot_epochs_train)
        loss_ax.legend()

        acc_ax.set_title("Accuracy of the train and validation sets")
        acc_ax.plot(plot_epochs_train, train_acc, label="Train Accuracy")
        acc_ax.plot(plot_epochs_val, val_acc, label="Validation Accuracy")
        acc_ax.set_xlabel("Epochs")
        acc_ax.set_ylabel("Percentage")
        acc_ax.set_xticks(plot_epochs_train)
        acc_ax.legend()

        if plot_file:
            plt.savefig(plot_file)

        if show:
            plt.show()
        else:
            plt.close()

        return ((train_loss[-1], train_acc[-1]), (val_loss[-1], val_acc[-1]))


    def compute_prediction(self, y):
        if y.shape[1] == 1:
            return y.int()

        return torch.max(y, 1)[1]

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
import seaborn as sns

from sklearn.metrics import accuracy_score

import utils


class SimpleLSTM(nn.Module):
    # Init
    def __init__(self, net_arch: dict):

        super(SimpleLSTM, self).__init__()

        input_size = net_arch["input_size"]
        hidden_size = net_arch["hidden_size"]
        num_layers = net_arch["num_layers"]
        lstm_dropout = net_arch["lstm_dropout"]

        assert len(net_arch["linear_features"]) == 1 + len(net_arch["linear_dropout"]),\
        f"Lenght of linear features ({len(net_arch['linear_features'])})"\
        f" must be one more than the the lenght of linear dropout"\
        f" ({len(net_arch['linear_dropout'])})."

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=lstm_dropout)

        # Layers - Linear
        self.dense = nn.ModuleList()
        self.ldrop = nn.ModuleList()

        # Flatten
        input_channels = hidden_size

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





    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:, -1]

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
                f.write(train_name + "\n")

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
        if y.shape[1] == 1:
            return y.int()

        return torch.max(y, 1)[1]

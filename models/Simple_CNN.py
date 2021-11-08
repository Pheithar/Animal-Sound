import torch
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    # Init
    def __init__(self, input_channels: int, input_shape: tuple, 
                 num_labels: int):

        super(SimpleCNN, self).__init__()
        # Fields
        self.input_channels = input_channels
        self.height, self.width = input_shape
        self.num_labels = num_labels

        # Layers - Conv
        self.conv1 = nn.Conv2d(3, 16, 2, padding="same")
        self.conv2 = nn.Conv2d(16, 32, 2, padding="same")
        self.conv3 = nn.Conv2d(32, 64, 2, padding="same")
        self.conv4 = nn.Conv2d(64, 32, 2, padding="same")
        self.conv5 = nn.Conv2d(32, 16, 2, padding="same")
        self.conv6 = nn.Conv2d(16, 8, 2, padding="same")
        
        # Layers - Linear
        self.dense1 = nn.Linear(8*16*4, 64)
        self.dense2 = nn.Linear(64, num_labels)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Activation
        self.relu = nn.ReLU()
        self.last_activation = nn.Softmax()

        # Flatten
        self.flatten = nn.Flatten()

        # Floaty type
        self.float()

    def forward(self, x_img):
        x = x_img

        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.relu(self.pool(self.conv4(x)))
        x = self.relu(self.pool(self.conv5(x)))

        x = self.relu(self.pool(self.conv6(x)))

        x = self.flatten(x)

        x = self.relu(self.dense1(x))
        x = self.last_activation(self.dense2(x))

        return x

    def train_loop(self, num_epochs: int, train_loader: DataLoader, 
                   validation_loader: DataLoader,
                   criterion, optimizer):

        self.train()
        for epoch in tqdm(range(num_epochs)):

            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # Get the inputs
                inputs, labels = data.values()
                
                # Make them variables
                optimizer.zero_grad()

                # Forward, backward and around
                outputs = self(inputs)
                print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data.item()
                if i % 1000 == 999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0
        print("Finished training")


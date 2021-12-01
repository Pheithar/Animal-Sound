from .Simple_CNN import SimpleCNN
from .Simple_LSTM import SimpleLSTM
from . import utils

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from tabulate import tabulate

class HierarchicalClassification():
    """HierarchicalClassification."""

    def __init__(self, nets_arch: dict, node_params: dict,
                 nets_organization: dict, dataset, random_seed: int = 42,
                 batch_size: int = 8, test_percentage: float = .2):
        """params:
        nets_arch: architecture of each of the network of the tree
        nets_organization: organization of the tree
        """
        self.organization = nets_organization
        self.architectures = nets_arch
        self.node_params = node_params
        self.root = self.build_tree(nets_organization)[0]
        self.nodes = self.get_nodes(self.root)

        self.dataset = dataset

        test_len = int((len(dataset) * test_percentage))
        train_len = len(dataset) - test_len

        train, test = random_split(dataset, [train_len, test_len],
                                   torch.Generator().manual_seed(random_seed))

        self.train_loader = DataLoader(train, batch_size=batch_size)
        self.test_loader = DataLoader(test, batch_size=batch_size)

    def build_tree(self, organization):
        nodes = []

        if isinstance(organization, dict):
            for key, value in organization.items():
                child_nodes = self.build_tree(value)

                node = HierarchicalNode(key, child_nodes, self.architectures[key],
                                        self.node_params[key])
                nodes.append(node)

            return nodes

        for node in organization:
            nodes.append(HierarchicalNode(node, [], self.architectures[node],
                                          self.node_params[node]))

        return nodes

    def get_nodes(self, root):
        nodes = [root]
        for node in root.childs:
            nodes.extend(self.get_nodes(node))

        return nodes

    def log_networks(self):
        for node in self.nodes:
            print(node.name)
            for child in node.net.children():
                print(child)
            print()

    def forward(self, x):
        # x = utils.get_cuda(x)
        outputs = {}
        for node in self.nodes:
            out = node.forward(x)
            outputs[node.name] = out
        return outputs

    def fit(self):
        for node in self.nodes:
            self.dataset.set_order(node.order)
            node.fit(self.train_loader, self.test_loader)

        train_acc = 0
        for x in self.train_loader:
            train_acc += self.predict(x)/len(self.train_loader)

        test_acc = 0
        for x in self.train_loader:
            test_acc += self.predict(x)/len(self.test_loader)

        return train_acc, test_acc


    def predict(self, x):

        acc = 0
        predictions = self.root.predict(x, self.dataset.labels).values()
        size = list(predictions)[0]["value"].shape[0]

        for i in range(size):
            max_value = 0
            label = None
            order = None
            for items in predictions:
                if items["value"][i] > max_value:
                    max_value = items["value"][i]
                    label = items["label"][i]
                    order = items["order"][-1]

            predict_class = self.dataset.order_hierarchy[order][label]
            predict_label = np.argmax(self.dataset.flatten_labels[predict_class])
            real_label = np.argmax(x["leaf_label"][i])

            if real_label == predict_label:
                acc += 1/size


        return acc

class HierarchicalNode():
    """docstring for HierarchicalNode."""

    def __init__(self, name: str, childs, net_arch: dict, params: dict):

        self.name = name
        self.childs = childs
        self.order = params["order"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.log_file = params["log_file"]
        self.img_file = params["img_file"]
        self.type = params["type"]

        if self.type == "CNN":
            self.net = utils.cuda_network(SimpleCNN(net_arch))
        if self.type == "LSTM":
            self.net = utils.cuda_network(SimpleLSTM(net_arch))

    def forward(self, x):
        if self.type == "CNN":
            x = x["image"]
        if self.type == "LSTM":
            x = x["mfcc"]

        x = utils.get_cuda(x)

        out = self.net(x)
        return utils.get_numpy(out)




    def fit(self, train_loader, test_loader):
        print(f"Training {self.name} network...")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr = self.lr)


        self.net.fit(self.epochs, train_loader, test_loader,
                     criterion, optimizer, frequency_val=10,
                     log_file=self.log_file, plot_file=self.img_file,
                     train_name=self.name, show=False)

        print("Done")

    def predict(self, x, labels, prediction={}, order=[], outputs={}):
        out = self.forward(x)

        for child in self.childs:
            child_name = child.name.split("_")[-1]
            index = np.argmax(labels[child_name])
            prediction[child_name] = out[:, index]

            child.predict(x, labels, prediction, order+[child_name], outputs)

        if not self.childs:
            mult = 1
            for key in order:
                mult *= prediction[key]
            value = mult * np.max(out, axis=1)
            label = np.argmax(out, axis=1)

            outputs["_".join(order)] = {}
            outputs["_".join(order)]["order"] = order
            outputs["_".join(order)]["label"] = label
            outputs["_".join(order)]["value"] = value

        return outputs

    def compute_prediction(self, y):
        return self.net.compute_prediction(y)

    def calculate_output(self, x, labels, value, output={}):
        with torch.no_grad():
            y = self.net(x)

        output[self.name] ={"value": utils.get_numpy(y)*value[:, None]}
        for child in self.childs:
            # child.calculate_output(x, value=)
            # print(child.name, y[:, i])
            # Get the one hot corresponding to the child
            one_hot = list(labels.get(child.name.split("_")[-1]))
            # Index of where the one is, because it is the same as the output
            idx = one_hot.index(1)
            new_value = utils.get_numpy(y[:, idx])*value
            # print(new_value)
            child.calculate_output(x, labels, value=new_value, output=output[self.name])

        return output

from Simple_CNN import SimpleCNN
import utils

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
        x = utils.get_cuda(x)
        outputs = {}
        for node in self.nodes:
            out = node.net(x)
            out = utils.get_numpy(out)

            outputs[node.name] = out
        return outputs

    def fit(self):
        for node in self.nodes:
            self.dataset.set_order(node.order)
            node.fit(self.train_loader, self.test_loader)



    def predict(self, x, verbose = False):
        x = utils.get_cuda(x)
        outputs = {}
        for node in self.nodes:
            out = node.net(x)
            y = node.predict(out)
            y = utils.get_numpy(y)

            if verbose:
                print(node.name)
                head = ["Label", "Probability"]
                tab = []
                for idx, o in zip(y, out):
                    tab.append([idx, f"{float(o[idx])*100:.2f}%"])
                print(tabulate(tab, head, tablefmt="github"))

            outputs[node.name] = y
        return outputs

    def calculate_output(self, x):
        value = np.ones(x.shape[0])
        x = utils.get_cuda(x)
        return self.root.calculate_output(x, self.dataset.labels, value)



class HierarchicalNode():
    """docstring for HierarchicalNode."""

    def __init__(self, name: str, childs, net_arch: dict, params: dict):

        self.name = name
        self.childs = childs
        self.order = params["order"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.log_file = params["log_file"]

        self.net = utils.cuda_network(SimpleCNN(net_arch))


    def fit(self, train_loader, test_loader):
        print(f"Training {self.name} network...")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr = self.lr)


        self.net.fit(self.epochs, train_loader, test_loader,
                     criterion, optimizer, frequency_val=5,
                     log_file=self.log_file, train_name=self.name)

        print("Done")

    def predict(self, y):
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



class HierarchicalClassification():
    """HierarchicalClassification."""

    def __init__(self, nets_arch: dict, net_organization: dict):
        """params:
        nets_arch: architecture of each of the network of the tree
        nets_organization: organization of the tree
        """
        self.organization = net_organization
        self.root_name = list(nets_organization.keys())[0]
        self.nodes = self.build_tree(nets_organization)

    def build_tree(self, organization):
        nodes = []

        if isinstance(organization, dict):
            for key, value in organization.items():
                child_nodes = self.build_tree(value)
                node = HierarchicalNode(key, child_nodes)
                nodes.append(node)
                if key == self.root_name:
                    self.root = node

            return nodes


        for node in organization:
            nodes.append(HierarchicalNode(node, []))

        return nodes

    def forward(self, x):
        pass

    def fit(self):
        for node in self.nodes:
            node.fit()

    def predict(self, x):
        y = self.root.predict(x)
        print(y)


class HierarchicalNode():
    """docstring for HierarchicalNode."""

    def __init__(self, name: str, childs):

        self.name = name
        self.childs = childs

    def fit(self):
        print(self.name)

        for child in self.childs:
            child.fit()

    def predict(self, x):
        x = x

        if len(self.childs) > 0:
            out = []
            for child in self.childs:
                out.append((child.name, child.predict(x)))
            return out
        return x

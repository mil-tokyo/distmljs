import torch.nn as nn


class NetMLP(nn.Module):
    def __init__(self, input_shape, n_classes, nch=256):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, nch)
        self.fc2 = nn.Linear(nch, nch)
        self.fc3 = nn.Linear(nch, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DoubleClippedNetMLP(nn.Module):
    def __init__(self, input_shape, n_classes, nch=256):
        super().__init__()
        self.q1 = NetMLP(input_shape, n_classes)
        self.q2 = NetMLP(input_shape, n_classes)

    def forward(self, x):
        y1 = self.q1(x)
        y2 = self.q2(x)

        return y1, y2

    def Q1(self, x):
        y = self.q1(x)
        return y


def make_net(name, input_shape, n_classes):
    if name == "mlp":
        return NetMLP(input_shape=input_shape, n_classes=n_classes)
    elif name == "dmlp":
        return DoubleClippedNetMLP(input_shape=input_shape, n_classes=n_classes)
    else:
        raise NotImplementedError(f"Unknown network: {name}")


def get_io_shape(dataset):
    if dataset == "cartpole":
        return 4, 2
    elif dataset == "maze":
        return 6, 4
    elif dataset == "doublePendulum":
        return 2+12, 2
    elif dataset == "Mujoco_InvertedDoublePendulum":
        return 11, 2
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

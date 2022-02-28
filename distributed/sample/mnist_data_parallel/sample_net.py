import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class NetMLP(nn.Module):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class NetConv(nn.Module):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def make_net(name, input_shape, n_classes):
    if name == "mlp":
        return NetMLP(input_shape, n_classes)
    elif name == "conv":
        return NetConv(input_shape, n_classes)
    else:
        raise NotImplementedError("Unknown net name")


def get_io_shape(dataset):
    if dataset == "mnist":
        return (1, 28, 28), 10
    elif dataset == "cifar10":
        return (3, 32, 32), 10
    elif dataset == "cifar100":
        return (3, 32, 32), 100
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")


def get_dataset_loader_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../pytorch_data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../pytorch_data', train=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader


def get_dataset_loader_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('../pytorch_data', train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10('../pytorch_data', train=False,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader


def get_dataset_loader_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR100('../pytorch_data', train=True, download=True,
                                      transform=transform)
    test_dataset = datasets.CIFAR100('../pytorch_data', train=False,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader


def get_dataset_loader(dataset):
    if dataset == "mnist":
        return get_dataset_loader_mnist()
    elif dataset == "cifar10":
        return get_dataset_loader_cifar10()
    elif dataset == "cifar100":
        return get_dataset_loader_cifar100()
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

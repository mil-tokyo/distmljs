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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, downsample):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if self.downsample:
            self.downsampleConv = nn.Conv2d(
                in_planes, planes, 1, stride=stride, bias=False)
            self.downsampleBN = nn.BatchNorm2d(planes)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        s = x
        if self.downsample:
            s = self.downsampleConv(s)
            s = self.downsampleBN(s)
        h = h + s
        h = F.relu(h)
        return h


class ResNet18(nn.Module):
    def __init__(self, input_shape, n_class):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_shape[0], 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 64, 2, 1)
        self.layer2 = self.make_layer(64, 128, 2, 2)
        self.layer3 = self.make_layer(128, 256, 2, 2)
        self.layer4 = self.make_layer(256, 512, 2, 2)
        self.fc = nn.Linear(512, n_class)
        self.flatten = nn.Flatten()

    def make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, stride != 1))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = F.adaptive_avg_pool2d(h, 1)
        h = self.flatten(h)
        h = self.fc(h)
        return h


def make_net(name, input_shape, n_classes):
    if name == "mlp":
        return NetMLP(input_shape, n_classes)
    elif name == "conv":
        return NetConv(input_shape, n_classes)
    elif name == "resnet18":
        return ResNet18(input_shape, n_classes)
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


def get_dataset_loader_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../pytorch_data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../pytorch_data', train=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_dataset_loader_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('../pytorch_data', train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10('../pytorch_data', train=False,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_dataset_loader_cifar100(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR100('../pytorch_data', train=True, download=True,
                                      transform=transform)
    test_dataset = datasets.CIFAR100('../pytorch_data', train=False,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_dataset_loader(dataset, batch_size):
    if dataset == "mnist":
        return get_dataset_loader_mnist(batch_size)
    elif dataset == "cifar10":
        return get_dataset_loader_cifar10(batch_size)
    elif dataset == "cifar100":
        return get_dataset_loader_cifar100(batch_size)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

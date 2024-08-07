"""
分散学習と同じモデルの学習をPyTorch単独で行う。
結果比較に用いる。
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sample_net import make_net


def train(model, loader, optimizer, device):
    model.train()
    losses = []
    for image, label in loader:
        optimizer.zero_grad()
        logit = model(image.to(device))
        loss = F.cross_entropy(logit, label.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for image, label in loader:
            label_d = label.to(device)
            logit = model(image.to(device))
            loss = F.cross_entropy(logit, label_d, reduction="sum")
            loss_sum += loss.item()
            pred = logit.argmax(dim=1, keepdim=True)
            correct += pred.eq(label_d.view_as(pred)).sum().item()
            count += len(image)

    return {"loss": loss_sum / count, "accuracy": correct / count}


def main():
    model_name = os.environ.get("MODEL", "conv")
    device = torch.device(os.environ.get("DEVICE", "cpu"))  # cuda
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(0)

    model = make_net(model_name)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('../pytorch_data', train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10('../pytorch_data', train=False,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    torch.save(model.state_dict(), os.path.join(
        output_dir, "initial_model.pt"))
    train_losses = []
    test_results = []
    for epoch in range(10):
        print(f"epoch {epoch}")
        train_losses.extend(train(model, train_loader, optimizer, device))
        test_result = test(model, test_loader, device)
        test_results.append(test_result)
        print(f"test: {test_result}")
    model.to("cpu")
    torch.save(model.state_dict(), os.path.join(
        output_dir, "pytorch_trained_model.pt"))
    with open(os.path.join(output_dir, "pytorch_training.pkl"), "wb") as f:
        pickle.dump({"train_losses": train_losses,
                    "test_results": test_results}, f)


if __name__ == "__main__":
    main()

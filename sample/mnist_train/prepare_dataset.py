import os
import numpy as np
from torchvision import datasets, transforms
from kakiage.tensor_serializer import serialize_tensors


def serialize_save(dataset, path):
    tostack = [[], []]
    for i in range(len(dataset)):
        data, target = dataset[i]
        tostack[0].append(data.numpy())
        tostack[1].append(np.int32(target))

    arrays = {
        "data": np.stack(tostack[0]).reshape(len(dataset), -1),
        "targets": np.stack(tostack[1]),
    }
    serialize_tensors(path, arrays)


def main():
    output_dir = os.path.join("output", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../pytorch_data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../pytorch_data', train=False,
                                  transform=transform)
    serialize_save(train_dataset, os.path.join(
        output_dir, "mnist_preprocessed_flatten_train.bin"))
    serialize_save(test_dataset, os.path.join(
        output_dir, "mnist_preprocessed_flatten_test.bin"))


if __name__ == "__main__":
    main()

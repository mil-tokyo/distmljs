import os
import numpy as np
from torchvision import datasets, transforms
from distmljs.tensor_serializer import serialize_tensors


def serialize_save(dataset, path, max_size=100000):
    tostack = [[], []]
    for i in range(min(len(dataset), max_size)):
        data, target = dataset[i]
        tostack[0].append(data.numpy())
        tostack[1].append(np.int32(target))

    arrays = {
        "data": np.stack(tostack[0]),
        "targets": np.stack(tostack[1]),
    }
    serialize_tensors(path, arrays)


def main():
    output_dir = os.path.join("output", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        "../pytorch_data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10("../pytorch_data", train=False, transform=transform)
    serialize_save(
        train_dataset, os.path.join(output_dir, "cifar10_preprocessed_train.bin"), 10000
    )
    serialize_save(
        test_dataset, os.path.join(output_dir, "cifar10_preprocessed_test.bin"), 1000
    )


if __name__ == "__main__":
    main()

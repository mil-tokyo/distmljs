# Sample: MNIST data parallel training

Training MLPs to classify MNIST image datasets in data-parallel distributed training

CIFAR10, CIFAR100 dataset can be also used.

# Build

```
npm install
npm run build
```

# Run training

Settings are made via environment variables.

- MODEL: one of mlp, conv, resnet18. Specify model type.
- N_CLIENTS: The number of clients participating in the distribution calculation, an integer greater than or equal to 1. If not specified, 1 is assumed to be specified.
- EPOCH: Number of learning epochs. Default is 2.
- BATCH_SIZE: Batch size. Total for all clients. Default is 32.

Execution is via uvicorn. Command sample (for Mac/Linux):

```
MODEL=conv N_CLIENTS=2 npm run train
```

On Windows, use the set command:

```
set MODEL=conv
set N_CLIENTS=2
npm run train
```

Open [http://localhost:8081/](http://localhost:8081/) with web browser. If you set `N_CLIENTS`, to run `N_CLIENTS` distributed clients, it must be opened in `N_CLIENTS` browser windows. Note: If three tabs are opened on one window, the computation speed of the tabs not displayed will be reduced.

The learned models are output in ONNX format and can be used for inference with WebDNN, ONNX Runtime Web, etc.

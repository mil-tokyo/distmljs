# Sample of training multi-layer perceptron for MNIST digit classification dataset

This sample downloads dataset from static HTTP server and trains within web browser. (no distributed training)

This sample describes how to

- prepare dataset
- construct a neural network model
- train and evaluate the model
- save and load the model

See comments in the source code.

# Build

```
npm install
npm run build
python prepare_dataset.py
```

# Run

```
cd ../..
npm run serve
```

Open [http://localhost:8080/sample/mnist_train/output/](http://localhost:8080/sample/mnist_train/output/) with web browser.

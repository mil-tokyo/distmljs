# Sample of training ResNet-18 image classifier model for CIFAR-10 image classification dataset

This sample downloads dataset from static HTTP server and trains within web browser. (no distributed training)

# Build

Build DistML.js itself first, at the project root.

```
cd ../..
npm run build
```

Then build this sample.

```
cd sample/resnet
npm install
npm run build
python prepare_dataset.py
```

# Run

```
cd ../..
npm run serve
```

Open [http://localhost:8080/sample/resnet/output/](http://localhost:8080/sample/resnet/output/) with web browser.

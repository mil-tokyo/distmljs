# Sample of training Transformer, the natural language modeling

This sample downloads dataset from static HTTP server and trains within web browser. (no distributed training)

# Build

```
npm install
npm run build
python convert_train_data.py
```

# Run

```
cd ../..
npm run serve
```

Open [http://localhost:8080/sample/transformer/output/](http://localhost:8080/sample/transformer/output/) with web browser.

# Gradient check

A sample to check if the gradient calculation matches the existing framework in a complex model.

```
python gradient_check.py
```

Open [http://localhost:8080/sample/transformer/output/gradient_check.html](http://localhost:8080/sample/transformer/output/gradient_check.html) with web browser.

# License

This sample is a port of the Transformer model and sample application from PyTorch. See `pytorch-license.txt` for the original license.

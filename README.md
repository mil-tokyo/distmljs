# DistML.js

DistML.js is the web browser-based deep learning library with distributed training tools.

# Features

- Multi-dimensional tensor
  - Acceleration with GPU
    - WebGL (WebGL2 only), WebGPU (experimental)
  - Useful tensor operations for pre / post processing
- Neural network building with define-by-run
  - All operators needed by ResNet are implemented
  - PyTorch-like API
- Template for distributed training server
  - Low-latency communication with WebSocket
  - Low-overhead tensor serialization
  - Implementation of data-parallel SGD

# Setup

node 16.x is needed.

```
npm install
```

## Python environment

Python 3.8+ is needed for dataset preprocessing in samples and distributed training feature. Dataset download and export of trained model to ONNX requires [PyTorch](https://pytorch.org/).

The installation of distributed training server library is described in [distributed](./distributed/).

# Build

## WebGPU shader

This commands are needed only when WebGPU shader is modified.

```
python tools/generate_webgputensor_glsl_unary_op.py
node tools/compile_webgpu_shader.js
```

## JavaScript (CommonJS)

CommonJS format to be loaded by application built with webpack. The output is generated in `dist` directory.

```
npm run build
```

Then run below to generate archive for distribution.

```
npm pack
```

`distmljs-<version>.tgz` is generated.

## JavaScript (Webpack)

Single file format for directly loading from HTML using `<script>` tag. It is generated to `webpack/distmljs.js`.

```
npm run webpack
```

# Test

DistML.js needs to unit test elements such as WebGL that do not work in node.js and have implementation differences between Web browsers.
For this reason, testing is performed on a Web browser using mocha.

## Build

```
npm run webpack:test
```

## Run

```
npm run serve
```

Open [http://localhost:8080/test/](http://localhost:8080/test/) with web browser. Test automatically starts and the result will be displayed.

# Samples

This section describes `scalar_regression` as an sample. For other samples, see `sample` directory.

## Build of DistML.js itself

```
npm run build
```

## Build of sample

```
cd sample/scalar_regression
npm install
npm run build
```

## Run

Run HTTP server

```
npm run serve
```

Open [http://localhost:8080/sample/scalar_regression/output/](http://localhost:8080/sample/scalar_regression/output/) with web browser.

# License

MIT

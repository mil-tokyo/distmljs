# DistML.js

DistML.js is the web browser-based deep learning library with distributed training tools.

# Features

- Multi-dimensional tensor
  - Acceleration with GPU
    - WebGL (WebGL2 only), WebGPU (experimental; `maxPool2d`, `avgPool2d` and
      `Tensor.minimum` / `maximum` / `equal` are not implemented for the
      WebGPU backend yet)
  - Useful tensor operations for pre / post processing
- Neural network building with define-by-run
  - All operators needed by ResNet are implemented
  - PyTorch-like API
- Template for distributed training server
  - Low-latency communication with WebSocket
  - Low-overhead tensor serialization
  - Implementation of data-parallel SGD

# Setup

node 20 or later is needed (verified with node 24.7).

```
npm install
```

Google Chrome (or Chromium) is needed to run the unit tests. If it is not
installed in a well-known location, set the `CHROME_PATH` environment variable.

## Python environment

Python 3.8+ is needed for dataset preprocessing in samples and distributed training feature. Dataset download and export of trained model to ONNX requires [PyTorch](https://pytorch.org/).

The installation of distributed training server library is described in [distributed](./distributed/).

# Build

## WebGPU shader

The shaders are written in WGSL. Those under `shader/webgpu/standard` are
written by hand and those under `shader/webgpu/autogen` are generated from the
templates in `tools` (the `autogen` directory is not checked in).
`tools/compile_webgpu_shader.js` bundles every `.wgsl` file into
`src/tensor/webgpu/shaders.ts`, which is checked in, so these commands are
needed only when a WebGPU shader is modified.

```
python tools/generate_webgputensor_wgsl_unary_op.py
python tools/generate_webgputensor_wgsl_binary_op.py
python tools/generate_webgputensor_wgsl_copy_op.py
python tools/generate_webgputensor_wgsl_reduction_op.py
node tools/compile_webgpu_shader.js
```

To compile every shader with the WGSL compiler of a browser and report errors:

```
node tools/validate_wgsl.mjs
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

## Run on a headless browser

```
npm test
```

This builds the test bundle and runs it on a headless Chrome, once per
backend. The process exits with a non-zero status if any test fails. WebGL and
WebGPU are provided by SwiftShader, so a GPU is not required, but a real
browser should be used for the final check.

Each backend can also be run on its own:

```
npm run webpack:test
npm run test:cpu
npm run test:webgl
npm run test:webgpu
npm run test:heavy
```

`heavy` is a slot for slow tests, but no test currently uses it, so it runs
the same set as `test:cpu`.

## Run on a browser manually

```
npm run webpack:test
npm run serve
```

Open [http://localhost:8080/test/](http://localhost:8080/test/) with web browser. Test automatically starts and the result will be displayed.
The backend to test is selected with the checkboxes at the top of the page.

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

Run HTTP server at the project root.

```
cd ../..
npm run serve
```

Open [http://localhost:8080/sample/scalar_regression/output/](http://localhost:8080/sample/scalar_regression/output/) with web browser.

# License

MIT

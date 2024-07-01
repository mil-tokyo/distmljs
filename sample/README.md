# DistML.js samples

- `hello` - describes how to load DistML.js and construct tensor using pure JavaScript.
- `scalar_regression` - basic TypeScript application which trains multi-layer perceptron with synthesized scalar data.
- `mnist_train` - trains multi-layer perceptron for MNIST digit classification dataset.
- `resnet` - trains ResNet-18 image classifier model for CIFAR-10 image classification dataset. This model is heavy and shows the usefulness of learning on GPUs.

# To use samples as application template

You can use these samples as the template for new application. As the samples references DistML.js by relative path, so you have to modify it if you copy a sample to other directory.

## Sample "hello"

This example loads `<PROJECT_ROOT>/webpack/distmljs.js` directly from `index.html`. You have to copy `distmljs.js` (or download from Github's release) to your project directory and modify `<script>` tag in `index.html`.

`distmljs.js` can be built by command `npm run webpack`.

## Samples using npm system

Other samples use npm and webpack for package dependency management. You have to install DistML.js as npm package in your application repository by `npm install /path/to/distmljs-<version>.tgz`.

`distmljs-<version>.tgz` can be built by command `npm run build && npm pack`.

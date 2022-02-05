import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';
import {
  assertFloat32R,
  shaderGenOutput,
  shaderGenTensorElementwiseGet,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformElementwise,
  shaderGenTensorOutputUniformElementwiseItem,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from './shaderHelper';

export function softmax(x: WebGLTensor): WebGLTensor {
  assertFloat32R([x], 'softmax');
  const dtype = x.dtype;
  if (x.ndim !== 2) {
    throw new Error('softmax: input must be 2dim');
  }

  const output = WebGLTensor.empty(x.shape, dtype);
  const ctx = getNNWebGLContext();
  const kernelName = `softmax_${dtype}_${x.shape[1]}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define K ${x.shape[1]}
${shaderGenTensorOutputUniform(2, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_x', 2, x.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(2, output.buffer.textureShape.dim)}
  float m = -1000.0;
  for (int i = 0; i < K; i++) {
    float t = get_tex_x(tex_output_0, i);
    if (t > m) {
      m = t;
    }
  }
  float sum = 0.0;
  for (int i = 0; i < K; i++) {
    float t = get_tex_x(tex_output_0, i);
    sum += exp(t - m);
  }
  float v = exp(get_tex_x(tex_output_0, tex_output_1) - m) / sum;
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_x' }], output, [
    ...shaderGenTensorOutputUniformItem(output),
    ...shaderGenTensorNDGetUniformItem('tex_x', x),
  ]);
  return output;
}

export function softmaxCrossEntropyBackward(
  softmax: WebGLTensor,
  label: WebGLTensor,
  gy: WebGLTensor
): WebGLTensor {
  if (softmax.dtype !== 'float32') {
    throw new Error('softmax must be float32');
  }
  if (label.dtype !== 'int32') {
    throw new Error('label must be int32');
  }
  if (gy.dtype !== 'float32') {
    throw new Error('gy must be float32');
  }
  if (
    softmax.buffer.dimPerPixel !== 1 &&
    label.buffer.dimPerPixel !== 1 &&
    gy.buffer.dimPerPixel !== 1
  ) {
    throw new Error('dimPerPixel must be 1');
  }
  const dtype = softmax.dtype;

  const output = WebGLTensor.empty(softmax.shape);
  const ctx = getNNWebGLContext();
  const kernelName = `softmaxCrossEntropyBackward`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(2, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet(
  'tex_softmax',
  2,
  softmax.buffer.textureShape.dim,
  softmax.dtype
)}
${shaderGenTensorNDGet(
  'tex_label',
  1,
  label.buffer.textureShape.dim,
  label.dtype
)}
${shaderGenTensorNDGet('tex_gy', 2, gy.buffer.textureShape.dim, gy.dtype)}
uniform float inv_batch;
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(2, output.buffer.textureShape.dim)}
  float v = get_tex_softmax(tex_output_0, tex_output_1);
  int label = get_tex_label(tex_output_0);
  if (tex_output_1 == label) {
    v -= 1.0;
  }
  v *= get_tex_gy(tex_output_0, tex_output_1) * inv_batch;
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: softmax, name: 'tex_softmax' },
      { tensor: label, name: 'tex_label' },
      { tensor: gy, name: 'tex_gy' },
    ],
    output,
    [
      ...shaderGenTensorOutputUniformItem(output),
      ...shaderGenTensorNDGetUniformItem('tex_softmax', softmax),
      ...shaderGenTensorNDGetUniformItem('tex_label', label),
      ...shaderGenTensorNDGetUniformItem('tex_gy', gy),
      { name: 'inv_batch', type: 'float', value: 1 / softmax.shape[0] },
    ]
  );
  return output;
}

export function nllLoss(x: WebGLTensor, label: WebGLTensor): WebGLTensor {
  if (x.dtype !== 'float32') {
    throw new Error('softmax must be float32');
  }
  if (label.dtype !== 'int32') {
    throw new Error('label must be int32');
  }
  if (x.buffer.dimPerPixel !== 1 && label.buffer.dimPerPixel !== 1) {
    throw new Error('dimPerPixel must be 1');
  }
  const dtype = x.dtype;

  const output = WebGLTensor.empty([]);
  const ctx = getNNWebGLContext();
  const kernelName = `nllLoss_${x.shape[0]}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(2, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_x', 2, x.buffer.textureShape.dim, x.dtype)}
${shaderGenTensorNDGet(
  'tex_label',
  1,
  label.buffer.textureShape.dim,
  label.dtype
)}
#define BATCH ${x.shape[0]}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(0, output.buffer.textureShape.dim)}
  float v = 0.0;
  for (int b = 0; b < BATCH; b++) {
    int label = get_tex_label(b);
    v += -log(get_tex_x(b, label));
  }
  v /= float(BATCH);
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: x, name: 'tex_x' },
      { tensor: label, name: 'tex_label' },
    ],
    output,
    [
      ...shaderGenTensorOutputUniformItem(output),
      ...shaderGenTensorNDGetUniformItem('tex_x', x),
      ...shaderGenTensorNDGetUniformItem('tex_label', label),
    ]
  );
  return output;
}

export function mseLoss(a: WebGLTensor, b: WebGLTensor): WebGLTensor {
  assertFloat32R([a, b], 'mseLoss');
  const dtype = a.dtype;

  const output = WebGLTensor.empty([]);
  const ctx = getNNWebGLContext();
  const kernelName = `mseLoss_${a.size}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(0, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_a', 1, a.buffer.textureShape.dim, a.dtype)}
${shaderGenTensorNDGet('tex_b', 1, b.buffer.textureShape.dim, b.dtype)}
#define SIZE ${a.size}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(0, output.buffer.textureShape.dim)}
  float v = 0.0;
  for (int b = 0; b < SIZE; b++) {
    float va = get_tex_a(b);
    float vb = get_tex_b(b);
    float diff = va - vb;
    v += diff * diff;
  }
  v /= float(SIZE);
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: a, name: 'tex_a' },
      { tensor: b, name: 'tex_b' },
    ],
    output,
    [
      ...shaderGenTensorOutputUniformItem(output),
      ...shaderGenTensorNDGetUniformItem('tex_a', a, [1]),
      ...shaderGenTensorNDGetUniformItem('tex_b', b, [1]),
    ]
  );
  return output;
}

export function mseLossBackprop(
  a: WebGLTensor,
  b: WebGLTensor,
  gy: WebGLTensor
): [WebGLTensor, WebGLTensor] {
  assertFloat32R([a, b, gy], 'mseLoss');
  const dtype = a.dtype;

  const outputa = WebGLTensor.empty(a.shape);
  const outputb = WebGLTensor.empty(a.shape);
  const ctx = getNNWebGLContext();
  const kernelName = `mseLossBackprop`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(1, outputa.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_a', 1, a.buffer.textureShape.dim, a.dtype)}
${shaderGenTensorNDGet('tex_b', 1, a.buffer.textureShape.dim, b.dtype)}
${shaderGenTensorNDGet('tex_gy', 1, gy.buffer.textureShape.dim, gy.dtype)}
uniform float coef;
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(1, outputa.buffer.textureShape.dim)}
  float v = (get_tex_a(tex_output_0) - get_tex_b(tex_output_0)) * get_tex_gy(0) * coef;
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: a, name: 'tex_a' },
      { tensor: b, name: 'tex_b' },
      { tensor: gy, name: 'tex_gy' },
    ],
    outputa,
    [
      ...shaderGenTensorOutputUniformItem(outputa),
      ...shaderGenTensorNDGetUniformItem('tex_a', a, [1]),
      ...shaderGenTensorNDGetUniformItem('tex_b', b, [1]),
      { name: 'coef', type: 'float', value: 2 / a.size },
    ]
  );
  const kernelNegName = 'mseLossBackpropNeg';
  if (!ctx.hasKernel(kernelNegName)) {
    ctx.addKernel(
      kernelNegName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniformElementwise(
  outputb.buffer.textureShape.dim,
  dtype
)}
${shaderGenTensorElementwiseGet(
  'tex_input',
  outputa.buffer.textureShape.dim,
  dtype
)}
void main() {
  float v = -get_tex_input();
  ${shaderGenOutput('v', dtype)}
}
`
    );
  }
  ctx.runKernel(
    kernelNegName,
    [{ tensor: outputa, name: 'tex_input' }],
    outputb,
    [...shaderGenTensorOutputUniformElementwiseItem(outputb)]
  );

  return [outputa, outputb];
}

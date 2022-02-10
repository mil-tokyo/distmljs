import { arrayEqual } from '../../../util';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

export function assertFloat32(
  tensors: WebGPUTensor[],
  functionName?: string
): void {
  for (const tensor of tensors) {
    if (tensor.dtype !== 'float32') {
      throw new Error(`${functionName}: input tensor must be float32`);
    }
  }
}

export function softmax(x: WebGPUTensor): WebGPUTensor {
  assertFloat32([x], 'softmax');
  if (x.ndim !== 2) {
    throw new Error('softmax: input must be 2dim');
  }
  const shaderName = 'softmax';

  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader, 3);
  }
  const output = WebGPUTensor.empty(x.shape);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: x.shape[0], type: 'uint32' },
    { value: x.shape[1], type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(x.shape[0], 4096) / 64), y: 1, z: 1 },
  });

  return output;
}

export function softmaxCrossEntropyBackward(
  softmax: WebGPUTensor,
  label: WebGPUTensor,
  gy: WebGPUTensor
): WebGPUTensor {
  assertFloat32([softmax, gy], 'softmaxCrossEntropyBackward');
  if (label.dtype !== 'int32') {
    throw new Error('label must be int32');
  }
  const shaderName = 'softmax_cross_entropy_backward';

  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader, 5);
  }
  const output = WebGPUTensor.empty(softmax.shape);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
    { value: softmax.shape[0], type: 'uint32' },
    { value: softmax.shape[1], type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [softmax, label, gy, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });

  return output;
}

export function nllLoss(x: WebGPUTensor, label: WebGPUTensor): WebGPUTensor {
  assertFloat32([x], 'nllLoss');
  if (label.dtype !== 'int32') {
    throw new Error('label must be int32');
  }

  const shaderName = `nll_loss`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader, 4);
  }
  const output = WebGPUTensor.empty([], 'float32');
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: x.shape[0], type: 'uint32' },
    { value: x.shape[1], type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, label, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: 1, y: 1, z: 1 },
  });

  return output;
}

export function mseLoss(a: WebGPUTensor, b: WebGPUTensor): WebGPUTensor {
  assertFloat32([a, b], 'mseLoss');
  if (!arrayEqual(a.shape, b.shape)) {
    throw new Error('mseLoss: shape mismatch');
  }

  const shaderName = `mse_loss`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader, 4);
  }
  const output = WebGPUTensor.empty([], 'float32');
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: a.size, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [a, b, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: 1, y: 1, z: 1 },
  });

  return output;
}

export function mseLossBackprop(
  a: WebGPUTensor,
  b: WebGPUTensor,
  gy: WebGPUTensor
): [WebGPUTensor, WebGPUTensor] {
  assertFloat32([a, b, gy], 'mseLossBackprop');
  const shaderName = 'mse_loss_backprop';

  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader, 6);
  }
  const outputa = WebGPUTensor.empty(a.shape);
  const outputb = WebGPUTensor.empty(a.shape);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: outputa.size, type: 'uint32' },
    { value: 2 / a.size, type: 'float32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [a, b, gy, outputa, outputb],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(outputa.size, 4096) / 64), y: 1, z: 1 },
  });

  return [outputa, outputb];
}

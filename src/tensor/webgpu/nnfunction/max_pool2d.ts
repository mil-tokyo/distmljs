import { maxPool2DCalcShape } from '../../nnFunctionUtil';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

interface MaxPool2dParams {
  kernelSize: number | number[];
  stride: number | number[];
  padding: number | number[];
  dilation: number | number[];
  ceilMode: boolean;
}

function assertFloat32(tensors: WebGPUTensor[], functionName: string): void {
  for (const tensor of tensors) {
    if (tensor.dtype !== 'float32') {
      throw new Error(`${functionName}: input tensor must be float32`);
    }
  }
}

/**
 * 最大値とその位置を同時に求める。
 */
function maxPool2dCore(
  x: WebGPUTensor,
  params: MaxPool2dParams & { returnIndices: boolean | 'spatial' | 'flatten' }
): [WebGPUTensor, WebGPUTensor] {
  assertFloat32([x], 'max_pool2d');
  const {
    batch,
    dilations: [dilations0, dilations1],
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0, pads1],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = maxPool2DCalcShape(params, x.shape);
  const outputShape = [batch, ch, outShape0, outShape1];
  const output = WebGPUTensor.empty(outputShape, 'float32');
  const outputIdx = WebGPUTensor.empty(outputShape, 'int32');

  const shaderName = 'max_pool2d';
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${shaderName}: shader not found`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
    { value: inShape0, type: 'uint32' },
    { value: inShape1, type: 'uint32' },
    { value: outShape0, type: 'uint32' },
    { value: outShape1, type: 'uint32' },
    { value: kernelShape0, type: 'uint32' },
    { value: kernelShape1, type: 'uint32' },
    { value: strides0, type: 'uint32' },
    { value: strides1, type: 'uint32' },
    { value: pads0, type: 'uint32' },
    { value: pads1, type: 'uint32' },
    { value: dilations0, type: 'uint32' },
    { value: dilations1, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, output, outputIdx],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
  return [output, outputIdx];
}

export function max_pool2d_webgpu(
  x: WebGPUTensor,
  params: MaxPool2dParams & { returnIndices: false }
): WebGPUTensor {
  const [output, outputIdx] = maxPool2dCore(x, params);
  outputIdx.dispose();
  return output;
}

export function max_pool2d_with_indices_webgpu(
  x: WebGPUTensor,
  params: MaxPool2dParams & { returnIndices: true | 'spatial' | 'flatten' }
): WebGPUTensor[] {
  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
  return maxPool2dCore(x, params);
}

export function max_pool2d_backprop_webgpu(
  indices: WebGPUTensor,
  gy: WebGPUTensor,
  xShape: ReadonlyArray<number>,
  params: MaxPool2dParams & { returnIndices: true | 'spatial' | 'flatten' }
): WebGPUTensor {
  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
  assertFloat32([gy], 'max_pool2d_backprop');
  const [, , inShape0, inShape1] = xShape;
  const inSpLen = inShape0 * inShape1;
  const [, , outShape0, outShape1] = indices.shape;
  const outSpLen = outShape0 * outShape1;
  const gx = WebGPUTensor.empty(xShape, 'float32');

  const shaderName = 'max_pool2d_backprop';
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${shaderName}: shader not found`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: gx.size, type: 'uint32' },
    { value: inSpLen, type: 'uint32' },
    { value: outSpLen, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [gy, indices, gx],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(gx.size, 4096) / 64), y: 1, z: 1 },
  });
  return gx;
}

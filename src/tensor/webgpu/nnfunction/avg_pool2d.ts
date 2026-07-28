import { avgPool2DCalcShape } from '../../nnFunctionUtil';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

interface AvgPool2dParams {
  kernelSize: number | number[];
  stride: number | number[];
  padding: number | number[];
  ceilMode: boolean;
  countIncludePad: boolean;
  divisorOverride?: number;
}

// shaderのdivModeと対応する
const DIV_MODE_CONSTANT = 0;
const DIV_MODE_PAD = 1;
const DIV_MODE_IMAGE = 2;

export function avg_pool2d_webgpu(
  x: WebGPUTensor,
  params: AvgPool2dParams
): WebGPUTensor {
  if (x.dtype !== 'float32') {
    throw new Error('avg_pool2d: input tensor must be float32');
  }
  const {
    batch,
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0b, pads1b, pads0e, pads1e],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = avgPool2DCalcShape(params, x.shape);
  let multiplierConstant = 0.0;
  let divMode: number;
  if (params.divisorOverride) {
    multiplierConstant = 1 / params.divisorOverride;
    divMode = DIV_MODE_CONSTANT;
  } else if (params.countIncludePad) {
    divMode = DIV_MODE_PAD;
  } else {
    divMode = DIV_MODE_IMAGE;
  }

  const output = WebGPUTensor.empty(
    [batch, ch, outShape0, outShape1],
    'float32'
  );
  const shaderName = 'avg_pool2d';
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
    { value: pads0b, type: 'uint32' },
    { value: pads1b, type: 'uint32' },
    { value: pads0e, type: 'uint32' },
    { value: pads1e, type: 'uint32' },
    { value: divMode, type: 'uint32' },
    { value: multiplierConstant, type: 'float32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, output],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
  return output;
}

export function avg_pool2d_backprop_webgpu(
  gy: WebGPUTensor,
  xShape: ReadonlyArray<number>,
  params: AvgPool2dParams
): WebGPUTensor {
  const {
    batch,
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0b, pads1b, pads0e, pads1e],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = avgPool2DCalcShape(params, xShape);
  // currently implements only global average pooling
  if (!(
    kernelShape0 === inShape0 &&
    kernelShape1 === inShape1 &&
    pads0b === 0 &&
    pads0e === 0 &&
    pads1b === 0 &&
    pads1e === 0 &&
    strides0 === kernelShape0 &&
    strides1 === kernelShape1 &&
    outShape0 === 1 &&
    outShape1 === 1
  )) {
    throw new Error(
      'currently, avg_pool2d_backprop_webgpu only implements global average pooling'
    );
  }

  const gx = WebGPUTensor.empty([batch, ch, inShape0, inShape1], 'float32');
  const inSpLen = inShape0 * inShape1;
  const shaderName = 'avg_pool2d_backprop_global';
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
    {
      value: params.divisorOverride ? 1 / params.divisorOverride : 1 / inSpLen,
      type: 'float32',
    },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [gy, gx],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(gx.size, 4096) / 64), y: 1, z: 1 },
  });
  return gx;
}

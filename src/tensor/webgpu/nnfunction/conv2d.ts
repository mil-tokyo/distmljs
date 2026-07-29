import { conv2DCalcShape } from '../../nnFunctionUtil';
import { bmm } from '../core/standard';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

interface Conv2dImplParams {
  stride: number | [number, number];
  padding: number | [number, number] | [number, number, number, number]; //TODO: support 'same' and 'valid'
  dilation: number | [number, number];
  groups: number;
}

/**
 * im2col / col2im のメタバッファは同一の構成
 */
function makeIm2colMeta(
  length: number,
  batch: number,
  group: number,
  chInPerGroup: number,
  inShape: number[],
  outShape: number[],
  kernelShape: number[],
  strides: number[],
  pads: number[],
  dilations: number[]
): WebGPUMetaBufferContentElement[] {
  return [
    { value: length, type: 'uint32' },
    { value: batch, type: 'uint32' },
    { value: group, type: 'uint32' },
    { value: chInPerGroup, type: 'uint32' },
    { value: inShape[0], type: 'uint32' },
    { value: inShape[1], type: 'uint32' },
    { value: outShape[0], type: 'uint32' },
    { value: outShape[1], type: 'uint32' },
    { value: kernelShape[0], type: 'uint32' },
    { value: kernelShape[1], type: 'uint32' },
    { value: strides[0], type: 'uint32' },
    { value: strides[1], type: 'uint32' },
    { value: pads[0], type: 'uint32' },
    { value: pads[1], type: 'uint32' },
    { value: dilations[0], type: 'uint32' },
    { value: dilations[1], type: 'uint32' },
  ];
}

function runIm2colKernel(
  shaderName: 'conv2d_im2col' | 'conv2d_col2im',
  input: WebGPUTensor,
  output: WebGPUTensor,
  meta: WebGPUMetaBufferContentElement[]
): void {
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${shaderName}: shader not found`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [input, output],
    meta: { elements: meta },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
}

function im2col(
  dX: WebGPUTensor,
  dI: WebGPUTensor,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  inShape: number[],
  outShape: number[],
  chInPerGroup: number
): void {
  runIm2colKernel(
    'conv2d_im2col',
    dX,
    dI,
    makeIm2colMeta(
      dI.size,
      batch,
      group,
      chInPerGroup,
      inShape,
      outShape,
      kernelShape,
      strides,
      pads,
      dilations
    )
  );
}

function col2im(
  dI: WebGPUTensor,
  dY: WebGPUTensor,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  inShape: number[],
  outShape: number[],
  chInPerGroup: number
): void {
  runIm2colKernel(
    'conv2d_col2im',
    dI,
    dY,
    makeIm2colMeta(
      dY.size,
      batch,
      group,
      chInPerGroup,
      inShape,
      outShape,
      kernelShape,
      strides,
      pads,
      dilations
    )
  );
}

export function conv2d_webgpu(
  x: WebGPUTensor,
  weight: WebGPUTensor,
  bias: WebGPUTensor | undefined,
  params: Conv2dImplParams
): WebGPUTensor {
  const {
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chInPerGroup,
    chOut,
    chOutPerGroup,
  } = conv2DCalcShape(params, x.shape, weight.shape);
  // TODO im2colが巨大になる場合に分割して実行
  const im2colData = WebGPUTensor.empty([
    group,
    batch,
    outShape[0],
    outShape[1],
    chInPerGroup,
    kernelShape[0],
    kernelShape[1],
  ]);
  im2col(
    x,
    im2colData,
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chInPerGroup
  );
  // im2colData(group, bout, cinkhkw) * weight(group, coutpergroup, cinkhkw) -> matmulData(group, bout, coutpergroup)
  const im2colDataRs = im2colData.reshape([
    group,
    batch * outShape[0] * outShape[1],
    chInPerGroup * kernelShape[0] * kernelShape[1],
  ]);
  im2colData.dispose();
  const weightRs = weight.reshape([
    group,
    chOutPerGroup,
    chInPerGroup * kernelShape[0] * kernelShape[1],
  ]);
  const matmulDataRs = bmm(im2colDataRs, weightRs, false, true);
  im2colDataRs.dispose();
  weightRs.dispose();
  const matmulData = matmulDataRs.reshape([
    group,
    batch,
    outShape[0],
    outShape[1],
    chOutPerGroup,
  ]);
  matmulDataRs.dispose();

  const yRs = matmulData.transpose([1, 0, 4, 2, 3]);
  matmulData.dispose();
  const y = yRs.reshape([batch, chOut, outShape[0], outShape[1]]);
  yRs.dispose();
  if (bias) {
    const biasRs = bias.reshape([1, -1, 1, 1]);
    const ybias = WebGPUTensor.add(y, biasRs);
    y.dispose();
    biasRs.dispose();
    return ybias;
  }
  return y;
}

export function conv2d_backprop_gb_webgpu(gy: WebGPUTensor): WebGPUTensor {
  return WebGPUTensor.sum(gy, [0, 2, 3]);
}

// TODO: extend to conv_transpose2d with minor change
export function conv2d_backprop_gxgw_webgpu(
  gy: WebGPUTensor,
  x: WebGPUTensor,
  weight: WebGPUTensor,
  skipGx: true,
  skipGw: false,
  params: Conv2dImplParams
): [null, WebGPUTensor];
export function conv2d_backprop_gxgw_webgpu(
  gy: WebGPUTensor,
  x: WebGPUTensor,
  weight: WebGPUTensor,
  skipGx: false,
  skipGw: true,
  params: Conv2dImplParams
): [WebGPUTensor, null];
export function conv2d_backprop_gxgw_webgpu(
  gy: WebGPUTensor,
  x: WebGPUTensor,
  weight: WebGPUTensor,
  skipGx: false,
  skipGw: false,
  params: Conv2dImplParams
): [WebGPUTensor, WebGPUTensor];
export function conv2d_backprop_gxgw_webgpu(
  gy: WebGPUTensor,
  x: WebGPUTensor,
  weight: WebGPUTensor,
  skipGx: boolean,
  skipGw: boolean,
  params: Conv2dImplParams
): [WebGPUTensor | null, WebGPUTensor | null] {
  const {
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chIn,
    chInPerGroup,
    chOut,
    chOutPerGroup,
  } = conv2DCalcShape(params, x.shape, weight.shape);
  // TODO im2colが巨大になる場合に分割して実行
  const gyg = gy.reshape([
    batch,
    group,
    chOutPerGroup,
    outShape[0],
    outShape[1],
  ]);
  const gyTransposeData = gyg.transpose([1, 0, 3, 4, 2]);
  gyg.dispose();

  let gw: WebGPUTensor | null = null;
  let gx: WebGPUTensor | null = null;
  if (!skipGw) {
    const im2colData = WebGPUTensor.empty([
      group,
      batch,
      outShape[0],
      outShape[1],
      chInPerGroup,
      kernelShape[0],
      kernelShape[1],
    ]);
    im2col(
      x,
      im2colData,
      batch,
      dilations,
      group,
      kernelShape,
      pads,
      strides,
      inShape,
      outShape,
      chInPerGroup
    );
    // dI(group, bout, cinkhkw) * dGyT(group, bout, coutpergroup) -> dgw(group, coutpergroup, cinkhkw)
    const im2colDataRs = im2colData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    im2colData.dispose();

    const gyTRs = gyTransposeData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);

    const gwRs = bmm(gyTRs, im2colDataRs, true, false);
    gyTRs.dispose();
    im2colDataRs.dispose();
    gw = gwRs.reshape([chOut, chInPerGroup, kernelShape[0], kernelShape[1]]);
    gwRs.dispose();
  }
  if (!skipGx) {
    // dGyT(group, bout, coutpergroup) * dW(group, coutpergroup, cinkhkw) -> dGi(group, bout, cinkhkw)
    const gyTRs = gyTransposeData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);
    const weightRs = weight.reshape([
      group,
      chOutPerGroup,
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    const matmul = bmm(gyTRs, weightRs, false, false);
    gyTRs.dispose();
    weightRs.dispose();
    const gxRs = WebGPUTensor.empty([
      batch,
      group,
      chInPerGroup,
      inShape[0],
      inShape[1],
    ]);
    col2im(
      matmul,
      gxRs,
      batch,
      dilations,
      group,
      kernelShape,
      pads,
      strides,
      inShape,
      outShape,
      chInPerGroup
    );
    matmul.dispose();
    gx = gxRs.reshape([batch, chIn, inShape[0], inShape[1]]);
    gxRs.dispose();
  }
  gyTransposeData.dispose();
  return [gx, gw];
}

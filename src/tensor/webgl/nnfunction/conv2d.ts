import { conv2DCalcShape } from '../../nnFunctionUtil';
import { getStride } from '../../shapeUtil';
import { bmm } from '../core/gemm';
import {
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from '../core/shaderHelper';
import { getNNWebGLContext, WebGLUniformItem } from '../webglContext';
import { WebGLTensor } from '../webglTensor';

interface Conv2dImplParams {
  stride: number | [number, number];
  padding: number | [number, number] | [number, number, number, number]; //TODO: support 'same' and 'valid'
  dilation: number | [number, number];
  groups: number;
}

export function conv2d_webgl(
  x: WebGLTensor,
  weight: WebGLTensor,
  bias: WebGLTensor | undefined,
  params: Conv2dImplParams
): WebGLTensor {
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
  const im2colData = WebGLTensor.empty([
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
    chIn,
    chInPerGroup
  );
  // im2colData(group, bout, cinkhkw) * weight(group, coutpergroup, cinkhkw) -> matmulData(group, bout, coutpergroup)
  const im2colDataRs = WebGLTensor.reshape(im2colData, [
    group,
    batch * outShape[0] * outShape[1],
    chInPerGroup * kernelShape[0] * kernelShape[1],
  ]);
  im2colData.dispose();
  const weightRs = WebGLTensor.reshape(weight, [
    group,
    chOutPerGroup,
    chInPerGroup * kernelShape[0] * kernelShape[1],
  ]);
  const matmulDataRs = bmm(im2colDataRs, weightRs, false, true);
  im2colDataRs.dispose();
  weightRs.dispose();
  const matmulData = WebGLTensor.reshape(matmulDataRs, [
    group,
    batch,
    outShape[0],
    outShape[1],
    chOutPerGroup,
  ]);
  matmulDataRs.dispose();

  if (bias) {
    const yRs = WebGLTensor.transpose(matmulData, [1, 0, 4, 2, 3]);
    matmulData.dispose();
    const y = WebGLTensor.reshape(yRs, [
      batch,
      chOut,
      outShape[0],
      outShape[1],
    ]);
    yRs.dispose();
    const biasRs = WebGLTensor.reshape(bias, [1, -1, 1, 1]);
    const ybias = WebGLTensor.add(y, biasRs);
    y.dispose();
    biasRs.dispose();
    return ybias;
  } else {
    const yRs = WebGLTensor.transpose(matmulData, [1, 0, 4, 2, 3]);
    matmulData.dispose();
    const y = WebGLTensor.reshape(yRs, [
      batch,
      chOut,
      outShape[0],
      outShape[1],
    ]);
    yRs.dispose();
    return y;
  }
}

export function conv2d_backprop_gb_webgl(gy: WebGLTensor): WebGLTensor {
  return WebGLTensor.sum(gy, [0, 2, 3]);
}

// TODO: extend to conv_transpose2d with minor change
export function conv2d_backprop_gxgw_webgl(
  gy: WebGLTensor,
  x: WebGLTensor,
  w: WebGLTensor,
  skipGx: true,
  skipGw: false,
  params: Conv2dImplParams
): [null, WebGLTensor];
export function conv2d_backprop_gxgw_webgl(
  gy: WebGLTensor,
  x: WebGLTensor,
  w: WebGLTensor,
  skipGx: false,
  skipGw: true,
  params: Conv2dImplParams
): [WebGLTensor, null];
export function conv2d_backprop_gxgw_webgl(
  gy: WebGLTensor,
  x: WebGLTensor,
  w: WebGLTensor,
  skipGx: false,
  skipGw: false,
  params: Conv2dImplParams
): [WebGLTensor, WebGLTensor];
export function conv2d_backprop_gxgw_webgl(
  gy: WebGLTensor,
  x: WebGLTensor,
  w: WebGLTensor,
  skipGx: boolean,
  skipGw: boolean,
  params: Conv2dImplParams
): [WebGLTensor | null, WebGLTensor | null] {
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
  } = conv2DCalcShape(params, x.shape, w.shape);
  // TODO im2colが巨大になる場合に分割して実行
  const gyg = WebGLTensor.reshape(gy, [
    batch,
    group,
    chOutPerGroup,
    outShape[0],
    outShape[1],
  ]);
  const gyTransposeData = WebGLTensor.transpose(gyg, [1, 0, 3, 4, 2]);
  gyg.dispose();

  let gw: WebGLTensor | null = null;
  let gx: WebGLTensor | null = null;
  if (!skipGw) {
    const im2colData = WebGLTensor.empty([
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
      chIn,
      chInPerGroup
    );
    // dI(group, bout, cinkhkw) * dGyT(group, bout, coutpergroup) -> dgw(group, coutpergroup, cinkhkw)
    const im2colDataRs = WebGLTensor.reshape(im2colData, [
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    im2colData.dispose();

    const gyTRs = WebGLTensor.reshape(gyTransposeData, [
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);

    const gwRs = bmm(gyTRs, im2colDataRs, true, false);
    gyTRs.dispose();
    im2colDataRs.dispose();
    gw = WebGLTensor.reshape(gwRs, [
      chOut,
      chInPerGroup,
      kernelShape[0],
      kernelShape[1],
    ]);
    gwRs.dispose();
  }
  if (!skipGx) {
    // dGyT(group, bout, coutpergroup) * dW(group, coutpergroup, cinkhkw) -> dGi(group, bout, cinkhkw)
    const gyTRs = WebGLTensor.reshape(gyTransposeData, [
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);
    const weightRs = WebGLTensor.reshape(w, [
      group,
      chOutPerGroup,
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    const matmul = bmm(gyTRs, weightRs, false, false);
    gyTRs.dispose();
    weightRs.dispose();
    const matmulRs = WebGLTensor.reshape(matmul, [
      group,
      batch,
      outShape[0],
      outShape[1],
      chInPerGroup,
      kernelShape[0],
      kernelShape[1],
    ]);
    matmul.dispose();
    const gxRs = WebGLTensor.empty([
      batch,
      group,
      chInPerGroup,
      inShape[0],
      inShape[1],
    ]);
    col2im(
      matmulRs,
      gxRs,
      batch,
      dilations,
      group,
      kernelShape,
      pads,
      strides,
      outShape,
      inShape,
      chInPerGroup
    );
    matmulRs.dispose();
    gx = WebGLTensor.reshape(gxRs, [batch, chIn, inShape[0], inShape[1]]);
    gxRs.dispose();
  }
  gyTransposeData.dispose();
  return [gx, gw];
}

function im2col(
  dX: WebGLTensor,
  dI: WebGLTensor,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  inShape: number[],
  outShape: number[],
  chIn: number,
  chInPerGroup: number
): void {
  const ctx = getNNWebGLContext();
  const kernelName = `conv2d_im2col`;
  // dI: group,batch,outShape[0],outShape[1],chInPerGroup,kernelShape[0],kernelShape[1]
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
uniform int K0;
uniform int K1;
uniform int S0;
uniform int S1;
uniform int P0;
uniform int P1;
uniform int D0;
uniform int D1;
uniform int IS0;
uniform int IS1;
${shaderGenTensorOutputUniform(7, dI.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_x', 5, dX.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(7, dI.buffer.textureShape.dim)}
  
  int in0 = tex_output_2 * S0 - P0 + tex_output_5 * D0;
  int in1 = tex_output_3 * S1 - P1 + tex_output_6 * D1;
  float v = 0.0;
  if (in0 >= 0 && in0 < IS0 && in1 >= 0 && in1 < IS1) {
    v = get_tex_x(tex_output_1, tex_output_0, tex_output_4, in0, in1);
  }

  ${shaderGenOutput('v')};
}
`
    );
  }
  const shaderParams: WebGLUniformItem[] = [
    ...shaderGenTensorOutputUniformItem(dI),
    ...shaderGenTensorNDGetUniformItem(
      'tex_x',
      dX,
      getStride([batch, group, chInPerGroup, inShape[0], inShape[1]])
    ),
    { name: 'K0', value: kernelShape[0], type: 'int' },
    { name: 'K1', value: kernelShape[1], type: 'int' },
    { name: 'S0', value: strides[0], type: 'int' },
    { name: 'S1', value: strides[1], type: 'int' },
    { name: 'P0', value: pads[0], type: 'int' },
    { name: 'P1', value: pads[1], type: 'int' },
    { name: 'D0', value: dilations[0], type: 'int' },
    { name: 'D1', value: dilations[1], type: 'int' },
    { name: 'IS0', value: inShape[0], type: 'int' },
    { name: 'IS1', value: inShape[1], type: 'int' },
  ];
  ctx.runKernel(kernelName, [{ tensor: dX, name: 'tex_x' }], dI, shaderParams);
}

function col2im(
  dI: WebGLTensor,
  dY: WebGLTensor,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  outShape: number[],
  inShape: number[],
  chInPerGroup: number
): void {
  // in/outはconvのforward基準 (convtransposeとは逆)
  // dI: group, batch, inShape[0], inShape[1], chOutPerGroup, kernelShape[0], kernelShape[1]
  // dY: batch, group, chOutPerGroup, outShape[0], outShape[1]

  const ctx = getNNWebGLContext();
  const kernelName = `conv2d_col2im_${kernelShape[0]}_${kernelShape[1]}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define K0 ${kernelShape[0]}
#define K1 ${kernelShape[1]}
uniform int S0;
uniform int S1;
uniform int P0;
uniform int P1;
uniform int D0;
uniform int D1;
uniform int OS0;
uniform int OS1;
${shaderGenTensorOutputUniform(5, dY.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_di', 7, dI.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(5, dY.buffer.textureShape.dim)}
  
  int b = tex_output_0, g = tex_output_1, co = tex_output_2, o0 = tex_output_3, o1 = tex_output_4;
  float v = 0.0;
  for (int k0 = 0; k0 < K0; k0++) {
    for (int k1 = 0; k1 < K1; k1++) {
      int i0s = o0 + P0 - k0 * D0;
      int i1s = o1 + P1 - k1 * D1;
      int i0 = i0s / S0;
      if (i0s - i0 * S0 != 0 || i0 < 0 || i0 >= OS0) {
        continue;
      }
      int i1 = i1s / S1;
      if (i1s - i1 * S1 != 0 || i1 < 0 || i1 >= OS1) {
        continue;
      }
      v += get_tex_di(g, b, i0, i1, co, k0, k1);
    }
  }

  ${shaderGenOutput('v')};
}
`
    );
  }
  const shaderParams: WebGLUniformItem[] = [
    ...shaderGenTensorOutputUniformItem(dY),
    ...shaderGenTensorNDGetUniformItem('tex_di', dI),
    { name: 'S0', value: strides[0], type: 'int' },
    { name: 'S1', value: strides[1], type: 'int' },
    { name: 'P0', value: pads[0], type: 'int' },
    { name: 'P1', value: pads[1], type: 'int' },
    { name: 'D0', value: dilations[0], type: 'int' },
    { name: 'D1', value: dilations[1], type: 'int' },
    { name: 'OS0', value: outShape[0], type: 'int' },
    { name: 'OS1', value: outShape[1], type: 'int' },
  ];
  ctx.runKernel(kernelName, [{ tensor: dI, name: 'tex_di' }], dY, shaderParams);
}

import { avgPool2DCalcShape, maxPool2DCalcShape } from '../../nnFunctionUtil';
import {
  assertFloat32R,
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

export function avg_pool2d_webgl(
  x: WebGLTensor,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    ceilMode: boolean;
    countIncludePad: boolean;
    divisorOverride?: number;
  }
): WebGLTensor {
  assertFloat32R([x], 'avg_pool2d');
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
  const countIncludePad = params.countIncludePad;
  let divMode: 'constant' | 'pad' | 'image';
  if (params.divisorOverride) {
    multiplierConstant = 1 / params.divisorOverride;
    divMode = 'constant';
  } else if (countIncludePad) {
    divMode = 'pad';
  } else {
    divMode = 'image';
  }

  const output = WebGLTensor.empty([batch, ch, outShape0, outShape1]);
  const ctx = getNNWebGLContext();
  const kernelName = `avg_pool2d_${kernelShape0}_${kernelShape1}_${divMode}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define KS0 ${kernelShape0}
#define KS1 ${kernelShape1}
uniform int ST0;
uniform int ST1;
uniform int PA0B;
uniform int PA1B;
uniform int PA0E;
uniform int PA1E;
uniform int IS0;
uniform int IS1;
${divMode === 'constant' ? 'uniform float AREAMUL;' : ''}
${shaderGenTensorOutputUniform(4, output.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_x', 4, x.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(4, output.buffer.textureShape.dim)}
  float v = 0.0;
  ${['pad', 'image'].includes(divMode) ? 'float area = 0.0;' : ''}
  for (int k0 = 0; k0 < KS0; k0++) {
    for (int k1 = 0; k1 < KS1; k1++) {
      int in0 = tex_output_2 * ST0 - PA0B + k0;
      int in1 = tex_output_3 * ST1 - PA1B + k1;
      if (in0 >= 0 && in0 < IS0 && in1 >= 0 && in1 < IS1) {
        float iv = get_tex_x(tex_output_0, tex_output_1, in0, in1);
        v += iv;
        ${divMode === 'image' ? 'area += 1.0;' : ''}
      }
      ${
        divMode === 'pad'
          ? 'if (in0 < IS0 + PA0E && in1 < IS1 + PA1E) {area += 1.0;}'
          : ''
      }
    }
  }
  ${['pad', 'image'].includes(divMode) ? 'v /= area;' : 'v *= AREAMUL;'}
  ${shaderGenOutput('v')};
}
`
    );
  }
  const shaderParams: WebGLUniformItem[] = [
    ...shaderGenTensorOutputUniformItem(output),
    ...shaderGenTensorNDGetUniformItem('tex_x', x),
    { name: 'ST0', value: strides0, type: 'int' },
    { name: 'ST1', value: strides1, type: 'int' },
    { name: 'PA0B', value: pads0b, type: 'int' },
    { name: 'PA1B', value: pads1b, type: 'int' },
    { name: 'PA0E', value: pads0e, type: 'int' },
    { name: 'PA1E', value: pads1e, type: 'int' },
    { name: 'IS0', value: inShape0, type: 'int' },
    { name: 'IS1', value: inShape1, type: 'int' },
  ];
  if (divMode === 'constant') {
    shaderParams.push({
      name: 'AREAMUL',
      value: multiplierConstant,
      type: 'float',
    });
  }
  ctx.runKernel(
    kernelName,
    [{ tensor: x, name: 'tex_x' }],
    output,
    shaderParams
  );
  return output;
}

export function avg_pool2d_backprop_webgl(
  gy: WebGLTensor,
  xShape: ReadonlyArray<number>,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    ceilMode: boolean;
    countIncludePad: boolean;
    divisorOverride?: number;
  }
): WebGLTensor {
  throw new Error('not implemented');
}

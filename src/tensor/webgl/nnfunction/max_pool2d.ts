import { maxPool2DCalcShape } from '../../nnFunctionUtil';
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
import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';

export function max_pool2d_webgl(
  x: WebGLTensor,
  params: {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    returnIndices: false;
    ceilMode: boolean;
  }
): WebGLTensor {
  assertFloat32R([x], 'max_pool2d');
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
  const output = WebGLTensor.empty([batch, ch, outShape0, outShape1]);
  const ctx = getNNWebGLContext();
  const kernelName = `max_pool2d_${kernelShape0}_${kernelShape1}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define KS0 ${kernelShape0}
#define KS1 ${kernelShape1}
uniform int ST0;
uniform int ST1;
uniform int DI0;
uniform int DI1;
uniform int PA0;
uniform int PA1;
uniform int IS0;
uniform int IS1;
${shaderGenTensorOutputUniform(4, output.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_x', 4, x.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(4, output.buffer.textureShape.dim)}
  float v = -10000.0;
  for (int k0 = 0; k0 < KS0; k0++) {
    for (int k1 = 0; k1 < KS1; k1++) {
      int in0 = tex_output_2 * ST0 - PA0 + k0 * DI0;
      int in1 = tex_output_3 * ST1 - PA1 + k1 * DI1;
      if (in0 >= 0 && in0 < IS0 && in1 >= 0 && in1 < IS1) {
        float iv = get_tex_x(tex_output_0, tex_output_1, in0, in1);
        if (iv > v) {
          v = iv;
        }
      }
    }
  }
  ${shaderGenOutput('v')};
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_x' }], output, [
    ...shaderGenTensorOutputUniformItem(output),
    ...shaderGenTensorNDGetUniformItem('tex_x', x),
    { name: 'ST0', value: strides0, type: 'int' },
    { name: 'ST1', value: strides1, type: 'int' },
    { name: 'DI0', value: dilations0, type: 'int' },
    { name: 'DI1', value: dilations1, type: 'int' },
    { name: 'PA0', value: pads0, type: 'int' },
    { name: 'PA1', value: pads1, type: 'int' },
    { name: 'IS0', value: inShape0, type: 'int' },
    { name: 'IS1', value: inShape1, type: 'int' },
  ]);
  return output;
}

export function max_pool2d_with_indices_webgl(
  x: WebGLTensor,
  params: {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    returnIndices: true;
    ceilMode: boolean;
  }
): WebGLTensor[] {
  throw new Error('not implemented');
}

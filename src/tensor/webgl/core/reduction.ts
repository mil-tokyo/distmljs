import {
  getReductionByAxis,
  getReductionByBroadcastShape,
} from '../../shapeUtil';
import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';
import {
  getTypeForDType,
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from './shaderHelper';

function dispatchSum(
  x: WebGLTensor,
  y: WebGLTensor,
  toShape: number[],
  toShapeStrides: number[],
  reductionShape: number[],
  reductionStrides: number[]
): void {
  if (x.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error('dispatchSum: RGBA texture not yet supported');
  }
  const dtype = x.dtype;
  const { scalarType } = getTypeForDType(dtype);

  const ctx = getNNWebGLContext();
  const tex_input_args: string[] = [];
  for (let i = 0; i < toShape.length; i++) {
    tex_input_args.push(`tex_output_${i}`);
  }
  let loopRangeDef = '';
  let forLoopHead = '';
  let forLoopTail = '';
  for (let i = 0; i < reductionShape.length; i++) {
    loopRangeDef += `#define RED_${i} ${reductionShape[i]}\n`;
    forLoopHead += `for (int red_${i} = 0; red_${i} < RED_${i}; red_${i}++) {\n`;
    forLoopTail += '}';
    tex_input_args.push(`red_${i}`);
  }
  const kernelName = `sum_${toShape.length}_${reductionShape.length}_${dtype}_${y.buffer.textureShape.dim}_${x.buffer.textureShape.dim}_${reductionShape}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(
  toShape.length,
  y.buffer.textureShape.dim,
  dtype
)}
${shaderGenTensorNDGet(
  'tex_input',
  toShape.length + reductionShape.length,
  x.buffer.textureShape.dim,
  dtype
)}
${loopRangeDef}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(
    toShape.length,
    y.buffer.textureShape.dim
  )}
  ${scalarType} v = ${scalarType}(0);
  ${forLoopHead}
  v += get_tex_input(${tex_input_args.join(',')});
  ${forLoopTail}
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  const joinStrides = [...toShapeStrides, ...reductionStrides];
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_input' }], y, [
    ...shaderGenTensorOutputUniformItem(y, toShape),
    ...shaderGenTensorNDGetUniformItem('tex_input', x, joinStrides),
  ]);
}

export function sum(
  x: WebGLTensor,
  axis?: number | number[] | null,
  keepdims?: boolean
): WebGLTensor {
  const {
    toShape,
    toShapeStrides,
    toShapeKeepdims,
    reductionShape,
    reductionStrides,
  } = getReductionByAxis(x.shape, axis);
  const y = WebGLTensor.empty(keepdims ? toShapeKeepdims : toShape, x.dtype);
  dispatchSum(x, y, toShape, toShapeStrides, reductionShape, reductionStrides);

  return y;
}

export function sumTo(
  x: WebGLTensor,
  shape: ReadonlyArray<number>
): WebGLTensor {
  const {
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides,
  } = getReductionByBroadcastShape(x.shape, shape);
  const y = WebGLTensor.empty(shape, x.dtype);
  dispatchSum(
    x,
    y,
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides
  );

  return y;
}

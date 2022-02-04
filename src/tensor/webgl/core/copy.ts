import { arange } from '../../../util';
import { getBroadcastStride } from '../../shapeUtil';
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

export function broadcastTo(
  x: WebGLTensor,
  shape: ReadonlyArray<number>
): WebGLTensor {
  if (x.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error('broadcastTo: RGBA texture not yet supported');
  }
  const xStride = getBroadcastStride(x.shape, shape);
  const dtype = x.dtype;
  const y = WebGLTensor.empty(shape, dtype);
  const { scalarType } = getTypeForDType(dtype);

  const ctx = getNNWebGLContext();
  const ndim = shape.length;
  const get_expr = arange(ndim)
    .map((i) => `tex_output_${i}`)
    .join(',');
  const kernelName = `broadcastTo_${ndim}_${dtype}_${y.buffer.textureShape.dim}_${x.buffer.textureShape.dim}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(ndim, y.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_input', ndim, x.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(ndim, y.buffer.textureShape.dim)}
  ${scalarType} v = get_tex_input(${get_expr});
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_input' }], y, [
    ...shaderGenTensorOutputUniformItem(y, shape),
    ...shaderGenTensorNDGetUniformItem('tex_input', x, xStride),
  ]);
  return y;
}

export function stridedCopy(
  x: WebGLTensor,
  newShape: ReadonlyArray<number>,
  xStride: ReadonlyArray<number>
): WebGLTensor {
  if (x.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error('stridedCopy: RGBA texture not yet supported');
  }
  const dtype = x.dtype;
  const y = WebGLTensor.empty(newShape, dtype);
  const { scalarType } = getTypeForDType(dtype);

  const ctx = getNNWebGLContext();
  const ndim = newShape.length;
  const get_expr = arange(ndim)
    .map((i) => `tex_output_${i}`)
    .join(',');
  const kernelName = `stridedCopy_${ndim}_${dtype}_${y.buffer.textureShape.dim}_${x.buffer.textureShape.dim}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(ndim, y.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_input', ndim, x.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(ndim, y.buffer.textureShape.dim)}
  ${scalarType} v = get_tex_input(${get_expr});
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_input' }], y, [
    ...shaderGenTensorOutputUniformItem(y, newShape),
    ...shaderGenTensorNDGetUniformItem('tex_input', x, xStride),
  ]);
  return y;
}

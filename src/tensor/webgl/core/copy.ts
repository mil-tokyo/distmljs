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
  const xStride = getBroadcastStride(x.shape, shape);
  const dtype = x.dtype;
  const y = WebGLTensor.empty(shape, dtype);
  const { scalarType } = getTypeForDType(dtype);

  const ctx = getNNWebGLContext();
  const ndim = shape.length;
  let get_expr: string;
  switch (ndim) {
    case 0:
      get_expr = '';
      break;
    case 1:
      get_expr = 'tex_output_0';
      break;
    case 2:
      get_expr = 'tex_output_0, tex_output_1';
      break;
    case 3:
      get_expr = 'tex_output_0, tex_output_1, tex_output_2';
      break;
    case 4:
      get_expr = 'tex_output_0, tex_output_1, tex_output_2, tex_output_3';
      break;
    case 5:
      get_expr =
        'tex_output_0, tex_output_1, tex_output_2, tex_output_3, tex_output_4';
      break;
    case 6:
      get_expr =
        'tex_output_0, tex_output_1, tex_output_2, tex_output_3, tex_output_4, tex_output_5';
      break;
    default:
      throw new Error(`${name}: dimension ${ndim} is not yet supported`);
  }
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

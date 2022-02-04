import { DType } from '../../../dtype';
import { getMultiBroadcastShape } from '../../shapeUtil';
import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';
import {
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from './shaderHelper';

// TODO: broadcastがない場合の最適化
// TODO: 片側がスカラーの場合の最適化

function binaryWrap(
  lhs: WebGLTensor,
  rhs: WebGLTensor,
  name: string,
  exprs: { [T in DType]?: string }
): WebGLTensor {
  if (lhs.dtype !== rhs.dtype) {
    throw new Error(
      `${name}: dtype of lhs(${lhs.dtype}) !== rhs(${rhs.dtype})`
    );
  }
  const dtype = lhs.dtype;
  if (lhs.buffer.dimPerPixel !== 1 || rhs.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error();
  }
  let returnType: string;
  const expr = exprs[dtype];
  if (!expr) {
    throw new Error(`${name}: dtype ${dtype} is not supported`);
  }
  switch (dtype) {
    case 'float32':
      returnType = 'float';
      break;
    case 'int32':
      returnType = 'int';
      break;
    default:
      returnType = 'uint';
      break;
  }

  const { shape, allStrides } = getMultiBroadcastShape([lhs.shape, rhs.shape]);
  const output = WebGLTensor.empty(shape, dtype);
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
  const kernelName = `binary_${name}_${ndim}_${dtype}_${output.buffer.textureShape.dim}_${lhs.buffer.textureShape.dim}_${rhs.buffer.textureShape.dim}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniform(ndim, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_lhs', ndim, lhs.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_rhs', ndim, rhs.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(ndim, output.buffer.textureShape.dim)}
  ${returnType} v_l = get_tex_lhs(${get_expr});
  ${returnType} v_r = get_tex_rhs(${get_expr});
  ${expr}
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: lhs, name: 'tex_lhs' },
      { tensor: rhs, name: 'tex_rhs' },
    ],
    output,
    [
      ...shaderGenTensorOutputUniformItem(output, shape),
      ...shaderGenTensorNDGetUniformItem('tex_lhs', lhs, allStrides[0]),
      ...shaderGenTensorNDGetUniformItem('tex_rhs', rhs, allStrides[1]),
    ]
  );
  return output;
}

export function coreadd(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  return binaryWrap(lhs, rhs, 'add', {
    float32: 'float v = v_l + v_r;',
    int32: 'int v = v_l + v_r;',
    uint8: 'uint v = v_l + v_r;',
  });
}

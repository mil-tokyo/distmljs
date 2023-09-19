import { DType } from '../../../dtype';
import { arange } from '../../../util';
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
  const get_expr = arange(ndim)
    .map((i) => `tex_output_${i}`)
    .join(',');
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

export function coresub(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  return binaryWrap(lhs, rhs, 'sub', {
    float32: 'float v = v_l - v_r;',
    int32: 'int v = v_l - v_r;',
    uint8: 'uint v = v_l - v_r;',
  });
}

export function coremul(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  return binaryWrap(lhs, rhs, 'mul', {
    float32: 'float v = v_l * v_r;',
    int32: 'int v = v_l * v_r;',
    uint8: 'uint v = v_l * v_r;',
  });
}

export function corediv(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  return binaryWrap(lhs, rhs, 'div', {
    float32: 'float v = v_l / v_r;',
    int32: 'int v = v_l / v_r;',
    uint8: 'uint v = v_l / v_r;',
  });
}

export function corepow(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  // pow(-1.5, 2) cases error in GLSL, but it is useful in normalization algorithm.
  // implementation: pow(abs(-1.5), 2)
  return binaryWrap(lhs, rhs, 'pow', {
    float32: 'float v = pow(abs(v_l), v_r);',
  });
}

export function sigmoidBackprop(
  lhs: WebGLTensor,
  rhs: WebGLTensor
): WebGLTensor {
  return binaryWrap(lhs, rhs, 'sigmoidBackprop', {
    float32: 'float v = (1.0 - v_l) * v_l * v_r;',
  });
}

export function tanhBackprop(
  lhs: WebGLTensor,
  rhs: WebGLTensor
): WebGLTensor {
  return binaryWrap(lhs, rhs, 'tanhBackprop', {
    float32: 'float v = (1.0 - v_l * v_l) * v_r;',
  });
}

export function reluBackprop(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  return binaryWrap(lhs, rhs, 'reluBackprop', {
    float32: 'float v = v_l > 0.0 ? v_r : 0.0;',
  });
}

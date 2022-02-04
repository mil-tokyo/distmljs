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

// TODO: SIMD等でのパフォーマンス改善
export function gemm(
  a: WebGLTensor,
  b: WebGLTensor,
  transa: boolean,
  transb: boolean
): WebGLTensor {
  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(`gemm: dtype of lhs(${a.dtype}) !== rhs(${b.dtype})`);
  }
  const dtype = a.dtype;
  if (a.buffer.dimPerPixel !== 1 || b.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error();
  }
  let m: number, n: number, k: number, bk: number;
  let stam: number, stak: number, stbk: number, stbn: number; //strides
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('must be 2dim');
  }
  if (transa) {
    [k, m] = a.shape;
    [stak, stam] = a.strides;
  } else {
    [m, k] = a.shape;
    [stam, stak] = a.strides;
  }
  if (transb) {
    [n, bk] = b.shape;
    [stbn, stbk] = b.strides;
  } else {
    [bk, n] = b.shape;
    [stbk, stbn] = b.strides;
  }
  if (k !== bk) {
    throw new Error('inner product length does not match');
  }

  const output = WebGLTensor.empty([m, n], dtype);
  const ctx = getNNWebGLContext();
  const kernelName = `gemm_${dtype}_${k}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define K ${k}
${shaderGenTensorOutputUniform(2, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_a', 2, a.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_b', 2, b.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(2, output.buffer.textureShape.dim)}
  float v = 0.0;
  for (int i = 0; i < K; i++) {
    v += get_tex_a(tex_output_0, i) * get_tex_b(i, tex_output_1);
  }
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: a, name: 'tex_a' },
      { tensor: b, name: 'tex_b' },
    ],
    output,
    [
      ...shaderGenTensorOutputUniformItem(output, [m, n]),
      ...shaderGenTensorNDGetUniformItem('tex_a', a, [stam, stak]),
      ...shaderGenTensorNDGetUniformItem('tex_b', b, [stbk, stbn]),
    ]
  );
  return output;
}

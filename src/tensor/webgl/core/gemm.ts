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

// batched gemm (b, m, k) * (b, k, n) -> (b, m, n)
export function bmm(
  a: WebGLTensor,
  b: WebGLTensor,
  transa: boolean,
  transb: boolean
): WebGLTensor {
  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(`bmm: dtype of lhs(${a.dtype}) !== rhs(${b.dtype})`);
  }
  const dtype = a.dtype;
  if (a.buffer.dimPerPixel !== 1 || b.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error();
  }
  let batch: number,
    m: number,
    n: number,
    k: number,
    bk: number,
    bbatch: number;
  let stab: number,
    stam: number,
    stak: number,
    stbb: number,
    stbk: number,
    stbn: number; //strides
  if (a.ndim !== 3 || b.ndim !== 3) {
    throw new Error('must be 3dim');
  }
  if (transa) {
    [batch, k, m] = a.shape;
    [stab, stak, stam] = a.strides;
  } else {
    [batch, m, k] = a.shape;
    [stab, stam, stak] = a.strides;
  }
  if (transb) {
    [bbatch, n, bk] = b.shape;
    [stbb, stbn, stbk] = b.strides;
  } else {
    [bbatch, bk, n] = b.shape;
    [stbb, stbk, stbn] = b.strides;
  }
  if (k !== bk) {
    throw new Error('inner product length does not match');
  }
  if (batch !== bbatch) {
    throw new Error('batch length does not match');
  }

  const output = WebGLTensor.empty([batch, m, n], dtype);
  const ctx = getNNWebGLContext();
  const kernelName = `bmm_${dtype}_${k}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define K ${k}
${shaderGenTensorOutputUniform(3, output.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_a', 3, a.buffer.textureShape.dim, dtype)}
${shaderGenTensorNDGet('tex_b', 3, b.buffer.textureShape.dim, dtype)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(3, output.buffer.textureShape.dim)}
  float v = 0.0;
  for (int i = 0; i < K; i++) {
    v += get_tex_a(tex_output_0, tex_output_1, i) * get_tex_b(tex_output_0, i, tex_output_2);
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
      ...shaderGenTensorOutputUniformItem(output, [batch, m, n]),
      ...shaderGenTensorNDGetUniformItem('tex_a', a, [stab, stam, stak]),
      ...shaderGenTensorNDGetUniformItem('tex_b', b, [stbb, stbk, stbn]),
    ]
  );
  return output;
}

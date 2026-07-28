import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

export function gemm(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transa: boolean,
  transb: boolean
): WebGPUTensor {
  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error('gemm: input must be float32');
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

  const shaderName = `gemm`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error();
    }
    ctx.createPipeline(shaderName, shader);
  }
  const output = WebGPUTensor.empty([m, n], 'float32');
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: m, type: 'uint32' },
    { value: n, type: 'uint32' },
    { value: k, type: 'uint32' },
    { value: stam, type: 'uint32' },
    { value: stak, type: 'uint32' },
    { value: stbk, type: 'uint32' },
    { value: stbn, type: 'uint32' },
    { value: 1.0, type: 'float32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [a, b, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: 256 / 8, y: 256 / 8, z: 1 },
  });

  return output;
}

/**
 * バッチ付きの行列積。a(batch, m, k) * b(batch, k, n) -> y(batch, m, n)
 * transa/transbはストライドの入れ替えで表現するため、転置の実体化は不要。
 */
export function bmm(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transa: boolean,
  transb: boolean
): WebGPUTensor {
  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(`bmm: dtype of lhs(${a.dtype}) !== rhs(${b.dtype})`);
  }
  if (a.ndim !== 3 || b.ndim !== 3) {
    throw new Error('bmm: must be 3dim');
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
    throw new Error('bmm: inner product length does not match');
  }
  if (batch !== bbatch) {
    throw new Error('bmm: batch length does not match');
  }

  const shaderName = 'bmm';
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error('bmm: shader not found');
    }
    ctx.createPipeline(shaderName, shader);
  }
  const output = WebGPUTensor.empty([batch, m, n], 'float32');
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: m, type: 'uint32' },
    { value: n, type: 'uint32' },
    { value: k, type: 'uint32' },
    { value: stab, type: 'uint32' },
    { value: stam, type: 'uint32' },
    { value: stak, type: 'uint32' },
    { value: stbb, type: 'uint32' },
    { value: stbk, type: 'uint32' },
    { value: stbn, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [a, b, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: 256 / 8, y: 256 / 8, z: batch },
  });

  return output;
}

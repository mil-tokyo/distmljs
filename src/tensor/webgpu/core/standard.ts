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
    ctx.createPipeline(shaderName, shader, 4);
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

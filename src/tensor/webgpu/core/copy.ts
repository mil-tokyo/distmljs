import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

/**
 * output[i] = x[srcOffset + sum(dim_d * xStride_d)]
 * dim_d は出力の平坦なインデックス i を newShape で分解したもの。
 */
export function stridedCopy(
  x: WebGPUTensor,
  newShape: ReadonlyArray<number>,
  xStride: ReadonlyArray<number>,
  srcOffset = 0
): WebGPUTensor {
  const output = WebGPUTensor.empty(newShape, x.dtype);
  stridedCopyInto(x, output, xStride, srcOffset);
  return output;
}

/**
 * 出力テンソルを与える版のstridedCopy。
 */
export function stridedCopyInto(
  x: WebGPUTensor,
  output: WebGPUTensor,
  xStride: ReadonlyArray<number>,
  srcOffset = 0
): void {
  const dtype = x.dtype;
  const ndim = output.ndim;
  const shaderName = `strided_copy_${dtype}_${ndim}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`stridedCopy: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
    { value: srcOffset, type: 'uint32' },
  ];
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: output.shape[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: xStride[dim], type: 'uint32' });
  }
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
}

/**
 * output[dstOffset + sum(dim_d * yStride_d)] = x[i]
 * dim_d は入力の平坦なインデックス i を x.shape で分解したもの。
 * 出力の一部の領域のみを書き換えるため、outputは呼び出し側が用意する。
 */
export function stridedSet(
  x: WebGPUTensor,
  output: WebGPUTensor,
  yStride: ReadonlyArray<number>,
  dstOffset = 0
): void {
  const dtype = x.dtype;
  const ndim = x.ndim;
  const shaderName = `strided_set_${dtype}_${ndim}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`stridedSet: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: x.size, type: 'uint32' },
    { value: dstOffset, type: 'uint32' },
  ];
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: x.shape[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: yStride[dim], type: 'uint32' });
  }
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(x.size, 4096) / 64), y: 1, z: 1 },
  });
}

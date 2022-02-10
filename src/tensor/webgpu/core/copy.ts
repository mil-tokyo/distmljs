import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

export function stridedCopy(
  x: WebGPUTensor,
  newShape: ReadonlyArray<number>,
  xStride: ReadonlyArray<number>
): WebGPUTensor {
  const dtype = x.dtype;
  const ndim = newShape.length;
  const shaderName = `strided_copy_${dtype}_${ndim}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`stridedCopy: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader, 3);
  }
  const output = WebGPUTensor.empty(newShape, dtype);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
  ];
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: newShape[dim], type: 'uint32' });
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

  return output;
}

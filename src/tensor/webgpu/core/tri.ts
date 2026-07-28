import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

function triCore(
  input: WebGPUTensor,
  diagonal: number,
  sign: number
): WebGPUTensor {
  if (input.ndim !== 2) {
    // TODO: support
    throw new Error(`${sign > 0 ? 'tril' : 'triu'}: input dim must be 2`);
  }
  const dtype = input.dtype;
  const shaderName = `tri_${dtype}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`tri: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const output = WebGPUTensor.empty(input.shape, dtype);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
    { value: input.shape[1], type: 'uint32' },
    { value: diagonal, type: 'int32' },
    { value: sign, type: 'int32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [input, output],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
  return output;
}

export function tril(input: WebGPUTensor, diagonal = 0): WebGPUTensor {
  return triCore(input, diagonal, 1);
}

export function triu(input: WebGPUTensor, diagonal = 0): WebGPUTensor {
  return triCore(input, diagonal, -1);
}

import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

function getPipeline(shaderName: string) {
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${shaderName}: shader not found`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  return ctx;
}

export function embedding_webgpu(
  x: WebGPUTensor,
  weight: WebGPUTensor
): WebGPUTensor {
  if (x.dtype !== 'int32') {
    throw new Error('embedding: index tensor must be int32');
  }
  if (weight.dtype !== 'float32') {
    throw new Error('embedding: weight tensor must be float32');
  }
  const [, embeddingDim] = weight.shape;
  const output = WebGPUTensor.empty([...x.shape, embeddingDim], 'float32');

  const shaderName = 'embedding';
  const ctx = getPipeline(shaderName);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
    { value: embeddingDim, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, weight, output],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });
  return output;
}

export function embedding_backprop_webgpu(
  x: WebGPUTensor,
  gy: WebGPUTensor,
  numEmbeddings: number,
  embeddingDim: number
): WebGPUTensor {
  if (x.dtype !== 'int32') {
    throw new Error('embedding: index tensor must be int32');
  }
  // カーネルは加算で書き込むため0初期化が必要
  const output = WebGPUTensor.zeros([numEmbeddings, embeddingDim], 'float32');

  const shaderName = 'embedding_backprop';
  const ctx = getPipeline(shaderName);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: gy.size, type: 'uint32' },
    { value: embeddingDim, type: 'uint32' },
    { value: numEmbeddings, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, gy, output],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(gy.size, 4096) / 64), y: 1, z: 1 },
  });
  return output;
}

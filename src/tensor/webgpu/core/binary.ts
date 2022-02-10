import { getMultiBroadcastShape } from '../../shapeUtil';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

// TODO: broadcastがない場合の最適化
// TODO: 片側がスカラーの場合の最適化

function binaryWrap(
  lhs: WebGPUTensor,
  rhs: WebGPUTensor,
  name: string
): WebGPUTensor {
  if (lhs.dtype !== rhs.dtype) {
    throw new Error(
      `${name}: dtype of lhs(${lhs.dtype}) !== rhs(${rhs.dtype})`
    );
  }
  const dtype = lhs.dtype;
  const { shape: outShape, allStrides: inAllStrides } = getMultiBroadcastShape([
    lhs.shape,
    rhs.shape,
  ]);
  const ndim = outShape.length;
  const shaderName = `binary_${name}_${dtype}_${ndim}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${name}: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader, 4);
  }
  const output = WebGPUTensor.empty(outShape, dtype);
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: output.size, type: 'uint32' },
  ];
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: outShape[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: inAllStrides[0][dim], type: 'uint32' });
  }
  for (let dim = 0; dim < ndim; dim++) {
    metaElements.push({ value: inAllStrides[1][dim], type: 'uint32' });
  }
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [lhs, rhs, output],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(output.size, 4096) / 64), y: 1, z: 1 },
  });

  return output;
}

export function coreadd(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'add');
}

export function coresub(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'sub');
}

export function coremul(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'mul');
}

export function corediv(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'div');
}

export function corepow(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'pow');
}

export function coresigmoidBackprop(
  lhs: WebGPUTensor,
  rhs: WebGPUTensor
): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'sigmoidBackprop');
}

export function corereluBackprop(
  lhs: WebGPUTensor,
  rhs: WebGPUTensor
): WebGPUTensor {
  return binaryWrap(lhs, rhs, 'reluBackprop');
}

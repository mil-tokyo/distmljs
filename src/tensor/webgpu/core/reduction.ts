import {
  getReductionByAxis,
  getReductionByBroadcastShape,
} from '../../shapeUtil';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

function dispatchSum(
  x: WebGPUTensor,
  y: WebGPUTensor,
  toShape: number[],
  toShapeStrides: number[],
  reductionShape: number[],
  reductionStrides: number[]
): void {
  const dtype = x.dtype;
  const shaderName = `sum_${dtype}_${toShape.length}_${reductionShape.length}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(
        `dispatchSum: toShape.length==${toShape.length}, reductionShape.length==${reductionShape.length} is not supported`
      );
    }
    ctx.createPipeline(shaderName, shader, 3);
  }
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: y.size, type: 'uint32' },
  ];
  for (let dim = 0; dim < toShape.length; dim++) {
    metaElements.push({ value: toShape[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < toShapeStrides.length; dim++) {
    metaElements.push({ value: toShapeStrides[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < reductionShape.length; dim++) {
    metaElements.push({ value: reductionShape[dim], type: 'uint32' });
  }
  for (let dim = 0; dim < reductionStrides.length; dim++) {
    metaElements.push({ value: reductionStrides[dim], type: 'uint32' });
  }
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, y],
    meta: {
      elements: metaElements,
    },
    workGroups: { x: Math.ceil(Math.min(y.size, 4096) / 64), y: 1, z: 1 },
  });
}

export function sum(
  x: WebGPUTensor,
  axis?: number | number[] | null,
  keepdims?: boolean
): WebGPUTensor {
  const {
    toShape,
    toShapeStrides,
    toShapeKeepdims,
    reductionShape,
    reductionStrides,
  } = getReductionByAxis(x.shape, axis);
  const y = WebGPUTensor.empty(keepdims ? toShapeKeepdims : toShape, x.dtype);
  dispatchSum(x, y, toShape, toShapeStrides, reductionShape, reductionStrides);

  return y;
}

export function sumTo(
  x: WebGPUTensor,
  shape: ReadonlyArray<number>
): WebGPUTensor {
  const {
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides,
  } = getReductionByBroadcastShape(x.shape, shape);
  const y = WebGPUTensor.empty(shape, x.dtype);
  dispatchSum(
    x,
    y,
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides
  );

  return y;
}

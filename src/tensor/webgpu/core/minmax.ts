import { arrayProd } from '../../../util';
import { calcSqueeze } from '../../shapeUtil';
import { webgpuShaders } from '../shaders';
import {
  getNNWebGPUContext,
  WebGPUMetaBufferContentElement,
} from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

/**
 * 1軸に沿った最大値/最小値とそのインデックスを求める。
 * dimを指定しない場合は全要素を対象とし、インデックスは平坦化した位置を指す。
 */
function minmaxCore(
  input: WebGPUTensor,
  isMax: boolean,
  dim: number | undefined,
  keepdim: boolean,
  functionName: string
): { values: WebGPUTensor; indices: WebGPUTensor; outShape: number[] } {
  if (input.size === 0) {
    throw new Error(`${functionName}: input mustn't be empty`);
  }
  if (input.dtype !== 'float32') {
    // CPU実装は任意のdtypeを扱えるが、値をfloat32として返すため精度が落ちる。
    // WebGPUでは対応dtypeを明示的に制限する。
    throw new Error(`${functionName}: input must be float32`);
  }

  let redLength: number;
  let innerLength: number;
  let outShape: number[];
  if (dim == undefined) {
    redLength = input.size;
    innerLength = 1;
    outShape = [];
  } else {
    const d = dim < 0 ? dim + input.ndim : dim;
    if (d < 0 || d >= input.ndim) {
      throw new Error(`${functionName}: dim ${dim} is out of range`);
    }
    redLength = input.shape[d];
    innerLength = arrayProd(input.shape.slice(d + 1));
    const keepdimShape = input.shape.slice();
    keepdimShape[d] = 1;
    outShape = keepdim ? keepdimShape : calcSqueeze(keepdimShape, d);
  }

  const shaderName = 'reduce_minmax';
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${shaderName}: shader not found`);
    }
    ctx.createPipeline(shaderName, shader);
  }
  const values = WebGPUTensor.empty(outShape, 'float32');
  const indices = WebGPUTensor.empty(outShape, 'int32');
  const metaElements: WebGPUMetaBufferContentElement[] = [
    { value: values.size, type: 'uint32' },
    { value: redLength, type: 'uint32' },
    { value: innerLength, type: 'uint32' },
    { value: isMax ? 1 : 0, type: 'uint32' },
  ];
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [input, values, indices],
    meta: { elements: metaElements },
    workGroups: { x: Math.ceil(Math.min(values.size, 4096) / 64), y: 1, z: 1 },
  });
  return { values, indices, outShape };
}

export function max(input: WebGPUTensor): WebGPUTensor;
export function max(
  input: WebGPUTensor,
  dim: number,
  keepdim?: boolean
): [WebGPUTensor, WebGPUTensor];
export function max(
  input: WebGPUTensor,
  dim?: number,
  keepdim = false
): WebGPUTensor | [WebGPUTensor, WebGPUTensor] {
  const { values, indices } = minmaxCore(input, true, dim, keepdim, 'max');
  if (dim == undefined) {
    indices.dispose();
    return values;
  }
  return [values, indices];
}

export function min(input: WebGPUTensor): WebGPUTensor;
export function min(
  input: WebGPUTensor,
  dim: number,
  keepdim?: boolean
): [WebGPUTensor, WebGPUTensor];
export function min(
  input: WebGPUTensor,
  dim?: number,
  keepdim = false
): WebGPUTensor | [WebGPUTensor, WebGPUTensor] {
  const { values, indices } = minmaxCore(input, false, dim, keepdim, 'min');
  if (dim == undefined) {
    indices.dispose();
    return values;
  }
  return [values, indices];
}

export function argmax(
  input: WebGPUTensor,
  dim?: number,
  keepdim = false
): WebGPUTensor {
  const { values, indices } = minmaxCore(input, true, dim, keepdim, 'argmax');
  values.dispose();
  return indices;
}

export function argmin(
  input: WebGPUTensor,
  dim?: number,
  keepdim = false
): WebGPUTensor {
  const { values, indices } = minmaxCore(input, false, dim, keepdim, 'argmin');
  values.dispose();
  return indices;
}

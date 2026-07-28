import { arraySum } from '../../../util';
import { calcCatShape } from '../../shapeUtil';
import { WebGPUTensor } from '../webgpuTensor';
import { stridedCopy, stridedSet } from './copy';

export function cat(
  tensors: ReadonlyArray<WebGPUTensor>,
  axis = 0
): WebGPUTensor {
  const { axisOffsets, yShape, dtype } = calcCatShape(tensors, axis);
  const output = WebGPUTensor.empty(yShape, dtype);
  // 入力テンソルごとに、出力の該当領域へ書き込む
  for (let i = 0; i < tensors.length; i++) {
    stridedSet(
      tensors[i],
      output,
      output.strides,
      axisOffsets[i] * output.strides[axis]
    );
  }
  return output;
}

export function cat_backprop_webgpu(
  gy: WebGPUTensor,
  shapes: ReadonlyArray<ReadonlyArray<number>>,
  axis: number
): WebGPUTensor[] {
  const gxs: WebGPUTensor[] = [];
  let axisOffset = 0;
  for (const shape of shapes) {
    gxs.push(stridedCopy(gy, shape, gy.strides, axisOffset * gy.strides[axis]));
    axisOffset += shape[axis];
  }
  return gxs;
}

export function split(
  x: WebGPUTensor,
  split_size_or_sections: number | number[],
  dim = 0
): WebGPUTensor[] {
  const dimShape: number[] = []; //dim次元における変更後の大きさ
  if (typeof split_size_or_sections === 'number') {
    const size = split_size_or_sections;
    const num = Math.ceil(x.shape[dim] / size);
    for (let i = 0; i < num; ++i) {
      if (i < num - 1) {
        dimShape[i] = size;
      } else {
        dimShape[i] = x.shape[dim] - size * (num - 1);
      }
    }
  } else {
    if (arraySum(split_size_or_sections) != x.shape[dim]) {
      throw new Error('split: sum of split size and tensor size must match');
    }
    const num = split_size_or_sections.length;
    for (let i = 0; i < num; ++i) {
      dimShape[i] = split_size_or_sections[i];
    }
  }

  const yShapes: number[][] = [];
  for (let i = 0; i < dimShape.length; ++i) {
    const yShape = x.shape.slice();
    yShape[dim] = dimShape[i];
    yShapes.push(yShape);
  }

  return cat_backprop_webgpu(x, yShapes, dim);
}

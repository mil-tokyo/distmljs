import { arraySum } from '../../../util';
import { calcCatShape } from '../../shapeUtil';
import { WebGPUTensor } from '../webgpuTensor';
import { stridedCopy, stridedSet } from './copy';

/**
 * xのaxis番目の軸に、大きさcountでストライド0の軸を挿入して読み出し、
 * 元の軸と併合した結果を返す。beforeがtrueなら挿入した軸が外側になる。
 *
 * before=false は repeat (各要素を連続して繰り返す)、
 * before=true は tile (全体を繰り返す) に対応する。
 */
function expandAxis(
  x: WebGPUTensor,
  axis: number,
  count: number,
  before: boolean
): WebGPUTensor {
  const expandedShape = [...x.shape];
  const xStride = [...x.strides];
  const insertAt = before ? axis : axis + 1;
  expandedShape.splice(insertAt, 0, count);
  xStride.splice(insertAt, 0, 0);
  const expanded = stridedCopy(x, expandedShape, xStride);
  const mergedShape = [...x.shape];
  mergedShape[axis] = x.shape[axis] * count;
  const merged = expanded.reshape(mergedShape);
  expanded.dispose();
  return merged;
}

export function repeat(
  x: WebGPUTensor,
  repeats: ReadonlyArray<number> | number,
  axis?: number
): WebGPUTensor {
  if (axis == undefined) {
    if (typeof repeats !== 'number') {
      throw new Error('repeat: repeats must be number when axis === undefined');
    }
    // 平坦化して各要素を繰り返す
    const flat = x.reshape([x.size]);
    const y = expandAxis(flat, 0, repeats, false);
    flat.dispose();
    return y;
  }
  if (typeof repeats === 'number') {
    return expandAxis(x, axis, repeats, false);
  }
  if (repeats.length !== x.shape[axis]) {
    throw new Error(
      `repeat: length of repeats (${repeats.length}) must match the size of axis ${axis} (${x.shape[axis]})`
    );
  }
  // 繰り返し回数が要素ごとに異なる場合は、軸方向のスライスごとに処理して連結する
  const sliceShape = [...x.shape];
  sliceShape[axis] = 1;
  const parts: WebGPUTensor[] = [];
  try {
    for (let i = 0; i < repeats.length; i++) {
      if (repeats[i] <= 0) {
        continue;
      }
      const slice = stridedCopy(x, sliceShape, x.strides, i * x.strides[axis]);
      if (repeats[i] === 1) {
        parts.push(slice);
      } else {
        parts.push(expandAxis(slice, axis, repeats[i], false));
        slice.dispose();
      }
    }
    if (parts.length === 0) {
      throw new Error('repeat: the result must not be empty');
    }
    return cat(parts, axis);
  } finally {
    for (const part of parts) {
      part.dispose();
    }
  }
}

export function tile(
  x: WebGPUTensor,
  reps: ReadonlyArray<number> | number
): WebGPUTensor {
  const yreps = typeof reps === 'number' ? [reps] : [...reps];
  const yDim = Math.max(x.ndim, yreps.length);
  // numpyと同じく、短いほうを先頭に1を補って右詰めで対応させる
  while (yreps.length < yDim) {
    yreps.unshift(1);
  }
  const alignedShape = [...x.shape];
  while (alignedShape.length < yDim) {
    alignedShape.unshift(1);
  }

  let y = x.reshape(alignedShape);
  for (let d = 0; d < yDim; d++) {
    if (yreps[d] === 1) {
      continue;
    }
    const next = expandAxis(y, d, yreps[d], true);
    y.dispose();
    y = next;
  }
  if (y.buffer === x.buffer) {
    // 一度も繰り返しがない場合はコピーを返す
    const copied = y.copy();
    y.dispose();
    return copied;
  }
  return y;
}

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

export function chunk(
  x: WebGPUTensor,
  chunks: number,
  dim = 0
): WebGPUTensor[] {
  if (x.ndim === 0) {
    throw new Error('chunk: chunk expects at least a 1-dimensional tensor');
  }
  //1つあたりの大きさ
  const size = Math.ceil(x.shape[dim] / chunks);
  return split(x, size, dim);
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

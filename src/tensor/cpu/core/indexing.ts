import { Ellipsis, slice, Slice } from '../..';
import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor, IndexingArg } from '../cpuTensor';

// TODO: copy.tsと統合
function broadcastCopy0(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  xStrides: ReadonlyArray<number>,
  srcOffset: number
): void {
  dy[0] = dx[srcOffset];
}

function broadcastCopy1(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  srcOffset: number
): void {
  const [ts0] = toShape;
  const [xt0] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    dy[id++] = dx[i0 * xt0 + srcOffset];
  }
}

function broadcastCopy2(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  srcOffset: number
): void {
  const [ts0, ts1] = toShape;
  const [xt0, xt1] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      dy[id++] = dx[i0 * xt0 + i1 * xt1 + srcOffset];
    }
  }
}

function broadcastCopy3(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  srcOffset: number
): void {
  const [ts0, ts1, ts2] = toShape;
  const [xt0, xt1, xt2] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        dy[id++] = dx[i0 * xt0 + i1 * xt1 + i2 * xt2 + srcOffset];
      }
    }
  }
}

function broadcastCopy4(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  srcOffset: number
): void {
  const [ts0, ts1, ts2, ts3] = toShape;
  const [xt0, xt1, xt2, xt3] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          dy[id++] = dx[i0 * xt0 + i1 * xt1 + i2 * xt2 + i3 * xt3 + srcOffset];
        }
      }
    }
  }
}

function broadcastCopyND(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  srcOffset: number
) {
  switch (toShape.length) {
    case 0:
      broadcastCopy0(dx, dy, toShape, xStrides, srcOffset);
      break;
    case 1:
      broadcastCopy1(dx, dy, toShape, xStrides, srcOffset);
      break;
    case 2:
      broadcastCopy2(dx, dy, toShape, xStrides, srcOffset);
      break;
    case 3:
      broadcastCopy3(dx, dy, toShape, xStrides, srcOffset);
      break;
    case 4:
      broadcastCopy4(dx, dy, toShape, xStrides, srcOffset);
      break;
    default:
      throw new Error(
        `broadcastCopyND to dimension ${toShape.length} is not yet supported.`
      );
  }
}

export function gets(tensor: CPUTensor, idxs: IndexingArg[]): CPUTensor {
  const tShape = tensor.shape;
  const tStrides = tensor.strides;
  const tNdim = tensor.ndim;
  const idxsWoEllipsis = ellipsisToSlice(idxs, tNdim);
  const idxsWoNewAxis: (number | Slice)[] = [];
  const newAxisDims: number[] = [];
  const squeezeDims: number[] = [];
  for (let i = 0; i < idxsWoEllipsis.length; i++) {
    const idx = idxsWoEllipsis[i];
    if (idx == null) {
      newAxisDims.push(i);
    } else {
      idxsWoNewAxis.push(idx);
      if (typeof idx === 'number') {
        squeezeDims.push(i);
      }
    }
  }
  // idxsWoNewAxis.length === tShape.length
  // Sliceを正規化
  const vShape: number[] = [];
  const stepStrides: number[] = [];
  let stepOffsetSum = 0;
  for (let i = 0; i < tShape.length; i++) {
    const tLen = tShape[i];
    const idx = idxsWoNewAxis[i];
    let start: number, stop: number, step: number, vLen: number;
    if (typeof idx === 'number') {
      start = idx < 0 ? idx + tLen : idx;
      if (start < 0 || start >= tLen) {
        throw new Error(
          `index ${idx} for axis ${i} is out of range for shape ${tShape}`
        );
      }
      stop = start + 1;
      step = 1;
      vLen = 1;
    } else {
      step = idx.step || 1;
      if (step > 0) {
        if (idx.start == null) {
          start = 0;
        } else {
          start = idx.start;
          if (start < 0) {
            start += tLen;
          }
          if (start < 0) {
            start = 0;
          }
          if (start > tLen) {
            start = tLen;
          }
        }
        if (idx.stop == null) {
          stop = tLen;
        } else {
          stop = idx.stop;
          if (stop < 0) {
            stop += tLen;
          }
          if (stop < 0) {
            stop = 0;
          }
          if (stop > tLen) {
            stop = tLen;
          }
        }

        if (stop > start) {
          vLen = Math.ceil((stop - start) / step);
        } else {
          vLen = 0;
        }
      } else {
        if (idx.start == null) {
          start = tLen - 1;
        } else {
          start = idx.start;
          if (start < 0) {
            start += tLen;
          }
          if (start < 0) {
            start = 0;
          }
          if (start >= tLen) {
            start = tLen - 1;
          }
        }
        if (idx.stop == null) {
          // a=arange(10)に対して
          // a[::-1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
          // a[:0:-1] = [9, 8, 7, 6, 5, 4, 3, 2, 1]
          // a[:-1:-1] = a[9:9:-1] = []
          // a[:-2:-1] = a[9:8:-1] = [9]
          // a[1::-1] = [1, 0]
          // a[0::-1] = [0]
          // a[-1::-1] = a[9::-1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
          stop = -1;
        } else {
          stop = idx.stop;
          if (stop < 0) {
            stop += tLen;
          }
          if (stop < 0) {
            stop = 0;
          }
          if (stop > tLen) {
            stop = tLen;
          }
        }
        if (stop < start) {
          vLen = Math.ceil((stop - start) / step);
        } else {
          vLen = 0;
        }
      }
    }
    const stepStride = tStrides[i] * step;
    const stepOffset = tStrides[i] * start;
    vShape.push(vLen);
    stepStrides.push(stepStride);
    stepOffsetSum += stepOffset;
  }

  for (const axis of newAxisDims) {
    vShape.splice(axis, 0, 1);
    stepStrides.splice(axis, 0, 0);
  }
  for (let i = squeezeDims.length - 1; i >= 0; i--) {
    const axis = squeezeDims[i];
    vShape.splice(axis, 1);
    stepStrides.splice(axis, 1);
  }

  const y = CPUTensor.zeros(vShape, tensor.dtype);
  broadcastCopyND(
    tensor.buffer.data,
    y.buffer.data,
    vShape,
    stepStrides,
    stepOffsetSum
  );

  return y;
}

function ellipsisToSlice(
  idxs: IndexingArg[],
  tNdim: number
): (number | Slice | null)[] {
  // ellipsisをsliceに変換する
  let sliceCount = 0;
  let hasEllipsis = false;
  for (let i = 0; i < idxs.length; i++) {
    const idx = idxs[i];
    if (idx instanceof Ellipsis) {
      if (hasEllipsis) {
        throw new Error('Multiple Ellipsis found.');
      }
      hasEllipsis = true;
    } else if (idx instanceof Slice || typeof idx === 'number') {
      sliceCount++;
    } else if (idx != null) {
      throw new Error('index must be any of number, Slice, Ellipsis, null.');
    }
  }
  const ndimToFill = tNdim - sliceCount;
  if (ndimToFill < 0) {
    throw new Error('The number of index exceeds array dimensions.');
  }
  let idxsWoEllipsis: (number | Slice | null)[];
  if (hasEllipsis) {
    idxsWoEllipsis = [];
    for (let i = 0; i < idxs.length; i++) {
      const idx = idxs[i];
      if (idx instanceof Ellipsis) {
        for (let j = 0; j < ndimToFill; j++) {
          idxsWoEllipsis.push(slice());
        }
      } else {
        idxsWoEllipsis.push(idx);
      }
    }
  } else {
    // 最後にslice()を追加
    idxsWoEllipsis = [...idxs] as (number | Slice | null)[];
    for (let i = 0; i < ndimToFill; i++) {
      idxsWoEllipsis.push(slice());
    }
  }
  return idxsWoEllipsis;
}

export function sets(
  tensor: CPUTensor | number,
  value: CPUTensor,
  idxs: IndexingArg[]
): void {
  // TODO
  throw new Error('Not implemented');
}

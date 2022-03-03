import { TypedArrayTypes } from '../../../dtype';
import { arraySum } from '../../../util';
import { CPUTensor } from '../cpuTensor';

function repeatSub(
  dx: TypedArrayTypes,
  xShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  repeats: ReadonlyArray<number>,
  axis: number
): number[] {
  let dy: number[] = [];
  if (axis === 0) {
    for (let i = 0; i < xShape[0]; ++i) {
      for (let j = 0; j < repeats[i]; ++j) {
        dy = [...dy, ...dx.slice(i * xStrides[0], (i + 1) * xStrides[0])];
      }
    }
  } else {
    for (let i = 0; i < xShape[0]; ++i) {
      dy = [
        ...dy,
        ...repeatSub(
          dx.slice(i * xStrides[0], (i + 1) * xStrides[0]),
          xShape.slice(1),
          xStrides.slice(1),
          repeats,
          axis - 1
        ),
      ];
    }
  }
  return dy;
}

export function repeat(
  x: CPUTensor,
  repeats: ReadonlyArray<number> | number,
  axis?: number
): CPUTensor {
  if (axis == undefined) {
    //軸指定がないとき
    if (typeof repeats !== 'number') {
      throw new Error('repeat: repeats must be number when axis === undefined');
    }
    if (x.ndim === 0) {
      //xがスカラーのとき
      const y = CPUTensor.zeros([repeats], x.dtype);
      const dx = x.getBuffer().data;
      const dy = y.getBuffer().data;
      for (let i = 0; i < repeats; ++i) {
        dy[i] = dx[0];
      }
      return y;
    } else {
      //xが多次元配列のとき
      const y = CPUTensor.zeros([x.getBuffer().length * repeats], x.dtype);
      const dx = x.getBuffer().data;
      const dy = y.getBuffer().data;
      for (let i = 0; i < x.getBuffer().length; ++i) {
        for (let j = 0; j < repeats; ++j) {
          dy[i * repeats + j] = dx[i];
        }
      }
      return y;
    }
  } else {
    //軸指定があるとき
    if (typeof repeats === 'number') {
      //repeatsがnumber型のとき
      const yShape: number[] = [];
      for (let i = 0; i < x.shape.length; ++i) {
        if (i === axis) {
          yShape[i] = x.shape[i] * repeats;
        } else {
          yShape[i] = x.shape[i];
        }
      }
      const dx = x.getBuffer().data;
      const newRepeats: number[] = [];
      for (let i = 0; i < x.shape[axis]; ++i) {
        newRepeats.push(repeats);
      }
      const dy = repeatSub(dx, x.shape, x.strides, newRepeats, axis);
      const y = CPUTensor.fromArray(dy, yShape, x.dtype);
      return y;
    } else {
      //repeatsが配列の時
      const yShape: number[] = [];
      for (let i = 0; i < x.shape.length; ++i) {
        if (i === axis) {
          yShape[i] = arraySum(repeats);
        } else {
          yShape[i] = x.shape[i];
        }
      }
      const dx = x.getBuffer().data;
      const dy = repeatSub(dx, x.shape, x.strides, repeats, axis);
      const y = CPUTensor.fromArray(dy, yShape, x.dtype);
      return y;
    }
  }
}

export function tile(
  x: CPUTensor,
  reps: ReadonlyArray<number> | number
): CPUTensor {
  // TODO: implement
  throw new Error();
}

function division_ceil(x: number, y: number): number {
  const result = (x - (x % y)) / y;
  if (x % y == 0) return result;
  else return result + 1;
}

function chunkSub(
  dx: TypedArrayTypes,
  xShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  yShape: ReadonlyArray<number>,
  size: number,
  place: number,
  order: number,
  dim: number,
  axis: number
): number[] {
  let dy: number[] = [];
  let yplace = place;
  if (dim == axis) yplace += size * xStrides[dim] * order;
  if (axis == xShape.length - 1) {
    for (let i = 0; i < yShape[axis]; ++i) {
      dy.push(dx[yplace + i]);
    }
  } else {
    for (let i = 0; i < yShape[axis]; ++i) {
      dy = [
        ...dy,
        ...chunkSub(
          dx,
          xShape,
          xStrides,
          yShape,
          size,
          yplace + xStrides[axis] * i,
          order,
          dim,
          axis + 1
        ),
      ];
    }
  }
  return dy;
}

export function chunk(x: CPUTensor, chunks: number, dim?: number): CPUTensor[] {
  if (x.ndim == 0) {
    throw new Error('chunk: chunk expects at least a 1-dimensional tensor');
  }
  if (typeof dim == 'undefined') {
    dim = 0;
  }
  const result: CPUTensor[] = [];
  const size = division_ceil(x.shape[dim], chunks); //1つあたりの大きさ
  const num = division_ceil(x.shape[dim], size); //要素数
  const lastsize = x.shape[dim] - size * (num - 1);
  const yShape: number[] = [];
  const lastShape: number[] = [];
  for (let i = 0; i < x.shape.length; ++i) {
    if (i === dim) {
      yShape[i] = size;
      lastShape[i] = lastsize;
    } else {
      yShape[i] = x.shape[i];
      lastShape[i] = x.shape[i];
    }
  }
  const dx = x.getBuffer().data;
  for (let i = 0; i < num; ++i) {
    if (i < num - 1) {
      const dy = chunkSub(dx, x.shape, x.strides, yShape, size, 0, i, dim, 0);
      result[i] = CPUTensor.fromArray(dy, yShape, x.dtype);
    } else {
      const dy = chunkSub(
        dx,
        x.shape,
        x.strides,
        lastShape,
        size,
        0,
        i,
        dim,
        0
      );
      result[i] = CPUTensor.fromArray(dy, lastShape, x.dtype);
    }
  }
  return result;
}

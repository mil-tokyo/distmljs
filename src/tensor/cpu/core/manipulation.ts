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

function tileSub(
  dx: TypedArrayTypes,
  xShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  yreps: ReadonlyArray<number>,
  place: number,
  dim: number,
  axis: number
): number[] {
  let dy: number[] = [];
  const p = axis + yreps.length - dim;
  const q = axis + xStrides.length - dim;
  let len;
  if (p < 0) len = 1;
  else len = yreps[p];
  let xsh;
  let xst;
  if (q < 0) {
    xsh = 1;
    xst = 1;
  } else {
    xsh = xShape[q];
    xst = xStrides[q];
  }
  if (axis == dim - 1) {
    for (let i = 0; i < len * xsh; ++i) {
      dy.push(dx[place + (i % xsh)]);
    }
  } else {
    for (let i = 0; i < len * xsh; ++i) {
      dy = [
        ...dy,
        ...tileSub(
          dx,
          xShape,
          xStrides,
          yreps,
          place + (i % xsh) * xst,
          dim,
          axis + 1
        ),
      ];
    }
  }
  return dy;
}

export function tile(
  x: CPUTensor,
  reps: ReadonlyArray<number> | number
): CPUTensor {
  const yreps: number[] = [];
  if (typeof reps === 'number') {
    //reps がnumber型のとき
    yreps[0] = reps;
  } else {
    //reps が配列のとき
    for (let i = 0; i < reps.length; ++i) {
      yreps[i] = reps[i];
    }
  }
  const yShape: number[] = [];
  const yDim = Math.max(x.shape.length, yreps.length);
  for (let i = 0; i < yDim; ++i) {
    const ax = i + x.shape.length - yDim;
    const ar = i + yreps.length - yDim;
    if (ax < 0) {
      yShape[i] = yreps[ar];
    } else if (ar < 0) {
      yShape[i] = x.shape[ax];
    } else {
      yShape[i] = x.shape[ax] * yreps[ar];
    }
  }
  const dx = x.getBuffer().data;
  const dy = tileSub(dx, x.shape, x.strides, yreps, 0, yDim, 0);
  const y = CPUTensor.fromArray(dy, yShape, x.dtype);
  return y;
}

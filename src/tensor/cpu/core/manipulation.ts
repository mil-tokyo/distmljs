import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor } from '../cpuTensor';

export function repeat(
  x: CPUTensor,
  repeats: ReadonlyArray<number> | number,
  axis?: number
): CPUTensor {
  // TODO: implement
  throw new Error();
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

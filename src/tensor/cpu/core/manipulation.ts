import { DType, TypedArrayForDType, TypedArrayTypes } from '../../../dtype';
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

function catSub(
  arrays: Array<TypedArrayTypes>,
  shapes: ReadonlyArray<ReadonlyArray<number>>,
  strides: ReadonlyArray<ReadonlyArray<number>>,
  axis: number,
  dtype: DType,
  dim: number
): Array<number> {
  const y: Array<number> = [];
  if (dim === axis) {
    for (const array of arrays) {
      for (let i = 0; i < array.length; ++i) {
        y.push(array[i]);
      }
    }
  } else {
    for (let i = 0; i < shapes[0][dim]; ++i) {
      const nextArrays: Array<TypedArrayTypes> = [];
      for (let j = 0; j < arrays.length; ++j) {
        nextArrays[j] = new TypedArrayForDType[dtype](strides[j][dim]);
      }
      for (let j = 0; j < arrays.length; ++j) {
        for (let k = 0; k < strides[j][dim]; ++k) {
          nextArrays[j][k] = arrays[j][i * strides[j][dim] + k];
        }
      }
      const t = catSub(nextArrays, shapes, strides, axis, dtype, dim + 1);
      for (let j = 0; j < t.length; ++j) {
        y.push(t[j]);
      }
    }
  }
  return y;
}

export function cat(tensors: ReadonlyArray<CPUTensor>, axis = 0): CPUTensor {
  if (tensors.length === 0) {
    throw new Error('tensors must not be empty');
  }
  const ndim = tensors[0].ndim;
  const shape = tensors[0].shape;
  const dtype = tensors[0].dtype;
  for (const tensor of tensors) {
    if (tensor.ndim !== ndim) {
      throw new Error('all tensors must be the same dimension');
    }
    for (let i = 0; i < shape.length; ++i) {
      if (i !== axis) {
        if (tensor.shape[i] !== shape[i]) {
          throw new Error(
            'all tensors must have the same shape except in the concatenating dimension'
          );
        }
      }
    }
    if (tensor.dtype !== dtype) {
      throw new Error('all tensors must have the same dtype');
    }
  }
  if (axis >= ndim) {
    throw new Error('axis must be smaller than tensor dimension');
  }
  const arrays: Array<TypedArrayTypes> = [];
  for (let i = 0; i < tensors.length; ++i) {
    arrays[i] = tensors[i].getBuffer().data;
  }
  const shapes = [];
  const strides = [];
  for (let i = 0; i < tensors.length; ++i) {
    shapes[i] = tensors[i].shape;
    strides[i] = tensors[i].strides;
  }
  const dy = catSub(arrays, shapes, strides, axis, dtype, 0);
  const yShape: Array<number> = [];
  for (let i = 0; i < tensors[0].ndim; ++i) {
    if (i === axis) {
      yShape[i] = 0;
      for (const tensor of tensors) {
        yShape[i] += tensor.shape[i];
      }
    } else {
      yShape[i] = shape[i];
    }
  }
  const y = CPUTensor.fromArray(dy, yShape);
  return y;
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
  if (dim === axis) yplace += size * xStrides[dim] * order;
  if (axis === xShape.length - 1) {
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

export function chunk(x: CPUTensor, chunks: number, dim = 0): CPUTensor[] {
  if (x.ndim === 0) {
    throw new Error('chunk: chunk expects at least a 1-dimensional tensor');
  }
  const result: CPUTensor[] = [];
  const size = Math.ceil(x.shape[dim] / chunks); //1つあたりの大きさ
  const num = Math.ceil(x.shape[dim] / size); //要素数
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

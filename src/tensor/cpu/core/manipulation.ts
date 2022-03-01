import { assert } from 'chai';
import { DType, TypedArrayForDType, TypedArrayTypes } from '../../../dtype';
import { arange, arrayProd, arraySum } from '../../../util';
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

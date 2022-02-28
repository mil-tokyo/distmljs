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
  shape: ReadonlyArray<number>,
  strides: ReadonlyArray<number>,
  axis: number,
  dtype: DType
): Array<number> {
  const y: Array<number> = [];
  if (axis == 0) {
    for (const array of arrays) {
      for (let i = 0; i < array.length; ++i) {
        y.push(array[i]);
      }
    }
  } else {
    const nextShape = shape.slice(1);
    const nextStrides = strides.slice(1);
    const nextArraySize = arrayProd(nextShape);
    const stride0 = strides[0];
    for (let i = 0; i < shape[0]; ++i) {
      const nextArrays: Array<TypedArrayTypes> = [];
      for (let j = 0; j < arrays.length; ++j) {
        nextArrays[j] = new TypedArrayForDType[dtype](nextArraySize);
      }
      for (let j = 0; j < arrays.length; ++j) {
        for (let k = 0; k < stride0; ++k) {
          nextArrays[j][k] = arrays[j][i * stride0 + k];
        }
      }
      const t = catSub(nextArrays, nextShape, nextStrides, axis - 1, dtype);
      for (let j = 0; j < t.length; ++j) {
        y.push(t[j]);
      }
    }
  }
  return y;
}

function cat(tensors: ReadonlyArray<CPUTensor>, axis = 0): CPUTensor {
  if (tensors.length === 0) {
    throw new Error('tensors must not be empty');
  }
  const tensor0 = tensors[0];
  const ndim = tensor0.ndim;
  const shape = tensor0.shape;
  const dtype = tensor0.dtype;
  for (const tensor of tensors) {
    if (tensor.ndim !== ndim) {
      throw new Error('all tensors must be the same dimention');
    }
    for (let i = 0; i < shape.length; ++i) {
      if (tensor.shape[i] !== shape[i]) {
        throw new Error('all tensors must be the same shape');
      }
    }
    if (tensor.dtype !== dtype) {
      throw new Error('all tensors must have the same dtype');
    }
  }
  if (axis >= ndim) {
    throw new Error('axis must be smaller than tensor dimention');
  }
  const arrays: Array<TypedArrayTypes> = [];
  for (let i = 0; i < tensors.length; ++i) {
    arrays[i] = tensors[i].getBuffer().data;
  }
  const dy = catSub(arrays, shape, tensor0.strides, axis, dtype);
  const yShape: Array<number> = [];
  for (let i = 0; i < tensors[0].ndim; ++i) {
    if (i === axis) {
      yShape[i] = shape[i] * tensors.length;
    } else {
      yShape[i] = shape[i];
    }
  }
  const y = CPUTensor.fromArray(dy, yShape);
  return y;
}

let x1 = CPUTensor.fromArray([1, 2, 3, 4], [4]);
let x2 = CPUTensor.fromArray([5, 6, 7, 8], [4]);
let x3 = CPUTensor.fromArray([9, 10, 11, 12], [4]);
let y = cat([x1, x2, x3]);
assert.deepEqual(y.shape, [12]);
assert.deepEqual(y.toArray(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

x1 = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
x2 = CPUTensor.fromArray([5, 6, 7, 8], [2, 2]);
y = cat([x1, x2]);
assert.deepEqual(y.shape, [4, 2]);
assert.deepEqual(y.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);

x1 = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
x2 = CPUTensor.fromArray([5, 6, 7, 8], [2, 2]);
y = cat([x1, x2], 1);
assert.deepEqual(y.shape, [2, 4]);
assert.deepEqual(y.toArray(), [1, 2, 5, 6, 3, 4, 7, 8]);

x1 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
x2 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
x3 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
y = cat([x1, x2, x3], 0);
assert.deepEqual(y.shape, [6, 3, 4]);
assert.deepEqual(y.get(3, 2, 2), 22);

x1 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
x2 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
x3 = CPUTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
y = cat([x1, x2, x3], 2);
assert.deepEqual(y.shape, [2, 3, 12]);
assert.deepEqual(y.get(1, 2, 8), 20);

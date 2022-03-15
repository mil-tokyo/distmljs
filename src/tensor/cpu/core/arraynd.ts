import { DType } from '../../../dtype';
import { ArrayNDValue, CPUTensor, NumberArrayND } from '../cpuTensor';

function isArrayLike(obj: unknown): obj is ArrayLike<ArrayNDValue> {
  if (obj && typeof obj === 'object') {
    if (typeof (obj as { length?: number }).length === 'number') {
      return true;
    }
  }
  return false;
}
function fromArrayNDSubShape(value: ArrayNDValue): number[] {
  let result: number[] = [];
  if (isArrayLike(value)) {
    const v0 = value[0];
    const resultSub = [...fromArrayNDSubShape(v0)];
    for (let i = 0; i < value.length; ++i) {
      const resultX = [...fromArrayNDSubShape(value[i])];
      for (let j = 0; j < resultSub.length; ++j) {
        if (resultSub[j] != resultX[j]) {
          throw new Error('size mismatch');
        }
        if (isArrayLike(v0)) {
          break;
        }
      }
    }
    result = [value.length, ...resultSub];
  } else if (value instanceof CPUTensor) {
    for (let i = 0; i < value.ndim; ++i) {
      result.push(value.shape[i]);
    }
  }
  return result;
}

function fromArrayNDSubElements(value: ArrayNDValue): number[] {
  const result: number[] = [];
  if (isArrayLike(value)) {
    for (let i = 0; i < value.length; ++i) {
      result.push(...fromArrayNDSubElements(value[i]));
    }
  } else if (value instanceof CPUTensor) {
    for (let i = 0; i < value.buffer.data.length; ++i) {
      result.push(value.buffer.data[i]);
    }
  } else {
    result.push(Number(value));
  }
  return result;
}

export function fromArrayND(value: ArrayNDValue, dtype?: DType): CPUTensor {
  if (value instanceof CPUTensor) {
    return value;
  }
  const shape = fromArrayNDSubShape(value);
  const elements = fromArrayNDSubElements(value);
  return CPUTensor.fromArray(elements, shape, dtype);
}

function toArrayNDSub(
  tensor: CPUTensor,
  place: number,
  axis: number
): NumberArrayND {
  const dim = tensor.ndim;
  const data = tensor.buffer.data;
  const stride = tensor.strides;
  const shape = tensor.shape;
  let array: NumberArrayND = [];
  if (axis === dim - 1) {
    for (let i = 0; i < shape[axis]; ++i) {
      array.push(data[place + i]);
    }
    return array;
  } else {
    for (let i = 0; i < shape[axis]; ++i) {
      array = [
        ...array,
        toArrayNDSub(tensor, place + i * stride[axis], axis + 1),
      ];
    }
    return array;
  }
}

export function toArrayND(tensor: CPUTensor): NumberArrayND {
  const buf = tensor.buffer.data;
  if (tensor.ndim === 0) {
    return buf[0];
  } else if (tensor.ndim === 1) {
    return Array.from(buf);
  } else {
    const result = toArrayNDSub(tensor, 0, 0);
    return result;
  }
}

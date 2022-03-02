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

export function fromArrayND(value: ArrayNDValue, dtype?: DType): CPUTensor {
  const elements: number[] = [];
  let shape: number[];
  if (value instanceof CPUTensor) {
    // TODO: ArrayLikeのさらに特殊ケースとしてCPUTensorがある。value.toArrayND()で通常のArrayLikeに変換してから扱うのと等価な結果を得たい
    throw new Error();
  } else if (isArrayLike(value)) {
    const v0 = value[0]; // TODO: value.length===0の場合
    if (isArrayLike(v0)) {
      const v00 = v0[0];
      if (isArrayLike(v00)) {
        throw new Error('3d case not implemented');
      } else {
        // 2d case
        shape = [value.length, v0.length];
        for (let i0 = 0; i0 < value.length; i0++) {
          // ArrayLikeはfor-ofに対応しているとは限らない
          const v = value[i0];
          if (!isArrayLike(v)) {
            throw new Error();
          }
          for (let i1 = 0; i1 < v.length; i1++) {
            elements.push(Number(v[i1]));
          }
        }
      }
    } else {
      // 1d case
      shape = [value.length];
      for (let i0 = 0; i0 < value.length; i0++) {
        elements.push(Number(value[i0]));
      }
    }
  } else {
    // 0d case
    elements.push(Number(value));
    shape = [];
  }

  return CPUTensor.fromArray(elements, shape, dtype);
}

export function toArrayND(tensor: CPUTensor): NumberArrayND {
  const buf = tensor.buffer.data;
  if (tensor.ndim === 0) {
    return buf[0];
  } else if (tensor.ndim === 1) {
    return Array.from(buf);
  } else {
    // TODO
    throw new Error('not implemented');
  }
}

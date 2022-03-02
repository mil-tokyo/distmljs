import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor } from '../cpuTensor';

function compareFuncAsc(a: number[], b: number[]): number {
  return a[0] - b[0];
}

function compareFuncDes(a: number[], b: number[]): number {
  return b[0] - a[0];
}

function sortSub(
  dx: TypedArrayTypes,
  shape: ReadonlyArray<number>,
  strides: ReadonlyArray<number>,
  dim: number,
  descending = false,
  curDim: number
): number[][] {
  const res: number[][] = [];
  if (curDim === dim) {
    for (let i = 0; i < strides[curDim]; ++i) {
      const nx: number[][] = [];
      for (let j = 0; j < shape[curDim]; ++j) {
        nx[j] = [dx[j * strides[curDim] + i], j];
      }
      if (descending) {
        nx.sort(compareFuncDes);
      } else {
        nx.sort(compareFuncAsc);
      }
      for (let j = 0; j < shape[curDim]; ++j) {
        res[j * strides[curDim] + i] = nx[j];
      }
    }
  } else {
    for (let i = 0; i < shape[curDim]; ++i) {
      const t = sortSub(
        dx.slice(i * strides[curDim], (i + 1) * strides[curDim]),
        shape,
        strides,
        dim,
        descending,
        curDim + 1
      );
      for (let j = 0; j < t.length; ++j) {
        res.push(t[j]);
      }
    }
  }
  return res;
}

export function sort(
  input: CPUTensor,
  dim = -1,
  descending = false
): [CPUTensor, CPUTensor] {
  if (dim < -input.ndim || dim >= input.ndim) {
    throw new Error(
      'A dim value within the range [-input.ndim, input.ndim) can be used'
    );
  }
  if (dim < 0) {
    dim = dim + input.ndim;
  }
  const dx = input.getBuffer().data;
  const dy = sortSub(dx, input.shape, input.strides, dim, descending, 0);
  const values = [];
  const indices = [];
  for (let i = 0; i < dy.length; ++i) {
    values[i] = dy[i][0];
    indices[i] = dy[i][1];
  }
  const y: [CPUTensor, CPUTensor] = [
    CPUTensor.fromArray(values, input.shape),
    CPUTensor.fromArray(indices, input.shape, 'int32'),
  ];
  return y;
}

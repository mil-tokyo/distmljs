import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor } from '../cpuTensor';
import { min } from './reduction';

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
    CPUTensor.fromArray(values, input.shape, input.dtype),
    CPUTensor.fromArray(indices, input.shape, 'int32'),
  ];
  return y;
}

function topkSub(
  dx: TypedArrayTypes,
  dxincides: TypedArrayTypes,
  k: number,
  dim: number,
  shape: ReadonlyArray<number>,
  strides: ReadonlyArray<number>,
  cur: number
): number[][] {
  if (cur === dim) {
    const res: number[][] = [];
    for (let i = 0; i < strides[cur]; ++i) {
      for (let j = 0; j < k; ++j) {
        res[i + j * strides[cur]] = [
          dx[i + j * strides[cur]],
          dxincides[i + j * strides[cur]],
        ];
      }
    }
    return res;
  } else {
    const res: number[][] = [];
    for (let i = 0; i < shape[cur]; ++i) {
      const t = topkSub(
        dx.slice(i * strides[cur], (i + 1) * strides[cur]),
        dxincides.slice(i * strides[cur], (i + 1) * strides[cur]),
        k,
        dim,
        shape,
        strides,
        cur + 1
      );
      for (let j = 0; j < t.length; ++j) {
        res.push(t[j]);
      }
    }
    return res;
  }
}

export function topk(
  input: CPUTensor,
  k: number,
  dim: number,
  largest: boolean
): [CPUTensor, CPUTensor] {
  if (k > input.shape[dim]) {
    throw new Error('k is out of range');
  }
  if (dim < -input.ndim || dim >= input.ndim) {
    throw new Error(
      'A dim value within the range [-input.ndim, input.ndim) can be used'
    );
  }
  if (dim < 0) {
    dim = dim + input.ndim;
  }
  const [x, xindices] = sort(input, dim, largest);
  const dx = x.getBuffer().data;
  const dxincides = xindices.getBuffer().data;
  const t = topkSub(dx, dxincides, k, dim, x.shape, x.strides, 0);
  const dy: number[] = [];
  const dyindices: number[] = [];
  for (let i = 0; i < t.length; ++i) {
    dy[i] = t[i][0];
    dyindices[i] = t[i][1];
  }
  const shape: number[] = [];
  for (let i = 0; i < x.ndim; ++i) {
    if (i === dim) {
      shape[i] = k;
    } else {
      shape[i] = x.shape[i];
    }
  }
  const y = CPUTensor.fromArray(dy, shape, x.dtype);
  const indices = CPUTensor.fromArray(dyindices, shape, 'int32');
  return [y, indices];
}

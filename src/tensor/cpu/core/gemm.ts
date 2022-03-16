import { CPUTensor } from '../cpuTensor';

/**
 * 転置込みの行列積を行う暫定的な関数
 * @param a
 * @param b
 * @param transa
 * @param transb
 */
export function gemm_cpu(
  a: CPUTensor,
  b: CPUTensor,
  transa: boolean,
  transb: boolean
): CPUTensor {
  let m: number, n: number, k: number, bk: number;
  let stam: number, stak: number, stbk: number, stbn: number; //strides
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('must be 2dim');
  }
  if (transa) {
    [k, m] = a.shape;
    [stak, stam] = a.strides;
  } else {
    [m, k] = a.shape;
    [stam, stak] = a.strides;
  }
  if (transb) {
    [n, bk] = b.shape;
    [stbn, stbk] = b.strides;
  } else {
    [bk, n] = b.shape;
    [stbk, stbn] = b.strides;
  }
  if (k !== bk) {
    throw new Error('inner product length does not match');
  }

  const output = CPUTensor.zeros([m, n]);
  let i = 0;
  const da = a.getBuffer().data;
  const db = b.getBuffer().data;
  const dy = output.getBuffer().data;
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0.0;
      for (let ip = 0; ip < k; ip++) {
        sum += da[row * stam + ip * stak] * db[col * stbn + ip * stbk];
      }
      dy[i++] = sum;
    }
  }
  return output;
}

// batched gemm (b, m, k) * (b, k, n) -> (b, m, n)
export function bmm_cpu(
  a: CPUTensor,
  b: CPUTensor,
  transa: boolean,
  transb: boolean
): CPUTensor {
  let batch: number,
    m: number,
    n: number,
    k: number,
    bk: number,
    bbatch: number;
  let stab: number,
    stam: number,
    stak: number,
    stbb: number,
    stbk: number,
    stbn: number; //strides
  if (a.ndim !== 3 || b.ndim !== 3) {
    throw new Error('must be 3dim');
  }
  if (transa) {
    [batch, k, m] = a.shape;
    [stab, stak, stam] = a.strides;
  } else {
    [batch, m, k] = a.shape;
    [stab, stam, stak] = a.strides;
  }
  if (transb) {
    [bbatch, n, bk] = b.shape;
    [stbb, stbn, stbk] = b.strides;
  } else {
    [bbatch, bk, n] = b.shape;
    [stbb, stbk, stbn] = b.strides;
  }
  if (k !== bk) {
    throw new Error('inner product length does not match');
  }
  if (batch !== bbatch) {
    throw new Error('batch length does not match');
  }

  const output = CPUTensor.zeros([batch, m, n]);
  let i = 0;
  const da = a.getBuffer().data;
  const db = b.getBuffer().data;
  const dy = output.getBuffer().data;
  for (let bb = 0; bb < batch; bb++) {
    const aofs = bb * stab;
    const bofs = bb * stbb;
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let sum = 0.0;
        for (let ip = 0; ip < k; ip++) {
          sum +=
            da[aofs + row * stam + ip * stak] *
            db[bofs + col * stbn + ip * stbk];
        }
        dy[i++] = sum;
      }
    }
  }
  return output;
}

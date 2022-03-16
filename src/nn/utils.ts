import { CPUTensor, Tensor } from '../tensor';
import { Parameter } from './core';

export function clipGradNorm_(
  params: Iterable<Parameter>,
  maxNorm: number,
  normType = 2
): Tensor {
  if (normType !== 2) {
    throw new Error('normType !== 2 is not yet implemented');
  }

  // currently CPU only
  const tensors: CPUTensor[] = [];
  for (const param of params) {
    const grad = param.grad?.data;
    if (!grad) {
      throw new Error();
    }
    if (!CPUTensor.isCPUTensor(grad)) {
      throw new Error('clipGradNorm_: currently CPUTensor only');
    }
    tensors.push(grad);
  }

  let sqsum = 0;
  for (const t of tensors) {
    const d = t.getBuffer().data;
    for (let i = 0; i < d.length; i++) {
      const v = d[i];
      sqsum += v * v;
    }
  }
  const norm = Math.sqrt(sqsum);
  const coef = maxNorm / (norm + 1e-6);
  if (coef < 1) {
    for (const t of tensors) {
      const d = t.getBuffer().data;
      for (let i = 0; i < d.length; i++) {
        d[i] *= coef;
      }
    }
  } // else {
  // do nothing (no amplification)
  //}

  return CPUTensor.s(norm);
}

import { CPUTensor } from '../cpuTensor';

function triCore(input: CPUTensor, diagonal: number, sign: number): CPUTensor {
  if (input.ndim !== 2) {
    // TODO: support
    throw new Error(`${sign > 0 ? 'tril' : 'triu'}: input dim must be 2`);
  }
  const dx = input.getBuffer().data;
  const y = CPUTensor.zeros(input.shape, input.dtype);
  const dy = y.getBuffer().data;
  const [shape0, shape1] = input.shape;
  const [stride0, stride1] = input.strides;

  for (let row = 0; row < shape0; row++) {
    for (let col = 0; col < shape1; col++) {
      let v: number;
      const idx = row * stride0 + col * stride1;
      if ((col - row - diagonal) * sign > 0) {
        v = 0;
      } else {
        v = dx[idx];
      }
      dy[idx] = v;
    }
  }
  return y;
}

export function tril(input: CPUTensor, diagonal = 0): CPUTensor {
  return triCore(input, diagonal, 1);
}

export function triu(input: CPUTensor, diagonal = 0): CPUTensor {
  return triCore(input, diagonal, -1);
}

import { Random } from '../../../math';
import { CPUTensor } from '../cpuTensor';

export function dropout_cpu(
  input: CPUTensor,
  p: number
): [CPUTensor, CPUTensor] {
  const rnd = Random.getDefault();
  const vec = rnd.random(input.size);
  const mask = CPUTensor.zeros(input.shape);
  const output = CPUTensor.zeros(input.shape);
  const dx = input.getBuffer().data;
  const dm = mask.getBuffer().data;
  const dy = output.getBuffer().data;
  const coef = 1.0 / (1.0 - p);
  for (let i = 0; i < input.size; i++) {
    if (vec[i] >= p) {
      dm[i] = coef;
      dy[i] = coef * dx[i];
    }
  }

  return [output, mask];
}

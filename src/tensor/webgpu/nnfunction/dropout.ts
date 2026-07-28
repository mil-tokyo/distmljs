import { Random } from '../../../math';
import { WebGPUTensor } from '../webgpuTensor';

/**
 * 乱数はCPU側のRandomで生成してGPUへ転送する。
 * GPU上での擬似乱数生成は実装しておらず、要素数分の転送が発生する。
 */
export function dropout_webgpu(
  input: WebGPUTensor,
  p: number
): [WebGPUTensor, WebGPUTensor] {
  if (input.dtype !== 'float32') {
    throw new Error('dropout: input tensor must be float32');
  }
  const rnd = Random.getDefault();
  const vec = rnd.random(input.size);
  const coef = 1.0 / (1.0 - p);
  const maskData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    if (vec[i] >= p) {
      maskData[i] = coef;
    }
  }
  const mask = WebGPUTensor.fromArray(maskData, input.shape);
  const output = WebGPUTensor.mul(input, mask);
  return [output, mask];
}

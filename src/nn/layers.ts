import { Variable } from '.';
import { CPUTensor } from '../tensor/cpu/cpuTensor';
import { Random } from '../math';
import { Layer, Parameter } from './core';
import { linear } from './functions';

export class Linear extends Layer {
  weight: Variable;
  bias?: Variable;

  constructor(
    public readonly inFeatures: number,
    public readonly outFeatures: number,
    bias = true
  ) {
    super();
    const random = Random.getDefault(); // TODO 指定可能にする
    // TODO: 分布を選択可能にする
    // uniform from [-sqrt(k), sqrt(k)] where k = 1 / in_features
    const rndscaled = (size: number, inFeatures: number) => {
      const sqrtK = Math.sqrt(1 / inFeatures);
      const ofs = -sqrtK;
      const scale = sqrtK * 2;
      const v = random.random(size); // [0, 1]
      for (let i = 0; i < size; i++) {
        v[i] = v[i] * scale + ofs;
      }
      return v;
    };
    this.weight = new Parameter(
      CPUTensor.fromArray(rndscaled(inFeatures * outFeatures, inFeatures), [
        outFeatures,
        inFeatures,
      ]),
      'weight'
    );
    if (bias) {
      this.bias = new Parameter(
        CPUTensor.fromArray(rndscaled(outFeatures, inFeatures), [outFeatures]),
        'bias'
      );
    }
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    return [await linear(inputs[0], this.weight, this.bias)];
  }
}

import { Variable } from '.';
import { CPUTensor } from '../tensor/cpu/cpuTensor';
import { Random } from '../math';
import { Layer, Parameter } from './core';
import { conv2d, Conv2dParams, linear } from './functions';

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

export interface Conv2dLayerParams extends Conv2dParams {
  bias?: boolean;
}

export class Conv2d extends Layer {
  readonly kernelSize: [number, number];
  weight: Variable;
  bias?: Variable;

  constructor(
    public readonly inChannels: number,
    public readonly outChannels: number,
    kernelSize: number | [number, number],
    public readonly params: Conv2dLayerParams
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
    if (typeof kernelSize === 'number') {
      this.kernelSize = [kernelSize, kernelSize];
    } else {
      if (kernelSize.length !== 2) {
        throw new Error('length of kernelSize is not 2');
      }
      this.kernelSize = [...kernelSize];
    }
    const chInPerGroup = this.inChannels / (params.groups || 1);
    this.weight = new Parameter(
      CPUTensor.fromArray(
        rndscaled(
          chInPerGroup * this.kernelSize[0] * this.kernelSize[1] * outChannels,
          chInPerGroup * this.kernelSize[0] * this.kernelSize[1]
        ),
        [outChannels, chInPerGroup, this.kernelSize[0], this.kernelSize[1]]
      ),
      'weight'
    );
    if (params.bias) {
      this.bias = new Parameter(
        CPUTensor.fromArray(rndscaled(outChannels, inChannels), [outChannels]),
        'bias'
      );
    }
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    return [await conv2d(inputs[0], this.weight, this.bias, this.params)];
  }
}

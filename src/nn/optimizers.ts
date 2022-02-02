import { CPUTensor } from '../tensor/cpu/cpuTensor';
import { Optimizer, Parameter } from './core';

export class SGD extends Optimizer {
  velocities: Map<Parameter, CPUTensor>;
  constructor(
    params: Iterable<Parameter>,
    public lr = 0.01,
    public momentum = 0.9
  ) {
    super(params);
    this.velocities = new Map();
  }

  async stepOne(parameter: Parameter): Promise<void> {
    let vel = this.velocities.get(parameter);
    if (!vel) {
      vel = CPUTensor.zeros(parameter.data.shape);
    }
    vel = CPUTensor.mul(vel, CPUTensor.s(this.momentum));
    vel = CPUTensor.add(
      vel,
      CPUTensor.mul(parameter.grad!.data as CPUTensor, CPUTensor.s(-this.lr))
    );

    parameter.data = CPUTensor.add(parameter.data as CPUTensor, vel);
    this.velocities.set(parameter, vel);
  }
}

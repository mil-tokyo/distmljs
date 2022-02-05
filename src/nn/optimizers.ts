import { Tensor } from '../tensor';
import { Optimizer, Parameter } from './core';

export class SGD extends Optimizer {
  velocities: Map<Parameter, Tensor>;
  constructor(
    params: Iterable<Parameter>,
    public lr = 0.01,
    public momentum = 0.9
  ) {
    super(params);
    this.velocities = new Map();
  }

  async stepOne(parameter: Parameter): Promise<void> {
    const T = parameter.data.getClass();
    let vel = this.velocities.get(parameter);
    if (!vel) {
      vel = T.zeros(parameter.data.shape);
    }
    // TODO: as anyを回避
    vel = T.mul(vel as any, T.s(this.momentum) as any);
    vel = T.add(
      vel as any,
      T.mul(parameter.grad!.data as any, T.s(-this.lr) as any) as any
    );

    parameter.data = T.add(parameter.data as any, vel as any);
    this.velocities.set(parameter, vel);
  }
}

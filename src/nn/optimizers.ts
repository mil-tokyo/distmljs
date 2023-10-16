import { tidy } from '..';
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
    const prevVel = this.velocities.get(parameter);
    const prevData = parameter.data;
    const [vel, newData] = await tidy(async () => {
      let vel = prevVel;
      if (!vel) {
        vel = T.zeros(parameter.data.shape);
      }
      // TODO: as anyを回避
      vel = T.mul(vel as any, this.momentum);
      vel = T.add(
        vel as any,
        T.mul(parameter.grad!.data as any, -this.lr) as any
      );
      const newData = T.add(prevData as any, vel as any);
      return [vel, newData];
    });

    parameter.data = newData;
    this.velocities.set(parameter, vel);
    if (prevVel) {
      // tidyでforward-backward-stepを囲む場合、forward前に存在するため保持されたままとなるパラメータを明示的に削除する必要
      prevData.dispose();
      prevVel.dispose();
    }
  }

  getKeepTensors(): Tensor[] {
    return [...this.velocities.values()];
  }
}

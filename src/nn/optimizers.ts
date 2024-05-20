import { tidy } from '..';
import { Tensor } from '../tensor';
import { Optimizer, Parameter } from './core';

export class SGD extends Optimizer {
  velocities: Map<Parameter, Tensor>;
  constructor(
    params: Iterable<Parameter>,
    public lr = 0.01,
    public momentum = 0.9,
    public weightDecay = 0
  ) {
    super(params);
    this.velocities = new Map();
  }

  async stepOne(parameter: Parameter): Promise<void> {
    const T = parameter.data.getClass();
    const prevVel = this.velocities.get(parameter);
    const prevData = parameter.data;
    const [vel, newData] = await tidy(async () => {
      // TODO: as anyを回避
      const grad = this.weightDecay ?
        T.add(
          parameter.grad!.data as any,
          T.mul(prevData as any, this.weightDecay) as any
        ) :
        parameter.grad!.data as any;
      let vel = prevVel;
      if (!vel) {
        vel = T.zeros(parameter.data.shape);
      }
      vel = T.mul(vel as any, this.momentum);
      vel = T.add(vel as any, T.mul(grad as any, -this.lr) as any);
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

export class Adam extends Optimizer {
  momentum: Map<Parameter, Tensor>;
  variance: Map<Parameter, Tensor>;
  deviceStateSteps: number;
  constructor(
    params: Iterable<Parameter>,
    public lr = 0.001,
    public beta1 = 0.9,
    public beta2 = 0.999,
    public weightDecay = 0,
    public epsilon = 1e-8
  ) {
    super(params);
    this.momentum = new Map();
    this.variance = new Map();
    this.deviceStateSteps = 0;
  }

  async step(): Promise<void> {
    await super.step();
    this.deviceStateSteps++;
  }

  async stepOne(parameter: Parameter): Promise<void> {
    const T = parameter.data.getClass();
    const prevMom = this.momentum.get(parameter);
    const prevVar = this.variance.get(parameter);
    const prevData = parameter.data;
    const [momentum, variance, newData] = await tidy(async () => {
      // TODO: as anyを回避
      const grad = this.weightDecay ?
        T.add(
          parameter.grad!.data as any,
          T.mul(prevData as any, this.weightDecay) as any
        ) :
        parameter.grad!.data as any;

      // momentum
      let m = prevMom || T.zeros(parameter.data.shape);
      m = T.add(
        T.mul(m as any, this.beta1) as any,
        T.mul(grad as any, 1 - this.beta1) as any
      );
      const mHat = T.div(
        m as any,
        1 - Math.pow(this.beta1, this.deviceStateSteps + 1)
      );

      // variance
      let v = prevVar || T.zeros(parameter.data.shape);
      v = T.add(
        T.mul(v as any, this.beta2) as any,
        T.mul(T.mul(grad as any, grad as any) as any, 1 - this.beta2) as any
      );
      const vHat = T.div(
        v as any,
        1 - Math.pow(this.beta2, this.deviceStateSteps + 1)
      );

      const newData = T.sub(
        prevData as any,
        T.mul(
          T.div(
            mHat as any,
            T.add(T.sqrt(vHat as any) as any, this.epsilon) as any
          ) as any,
          this.lr
        ) as any
      ) as any;

      return [m, v, newData];
    });

    parameter.data = newData;
    this.momentum.set(parameter, momentum);
    this.variance.set(parameter, variance);
    if (prevMom && prevVar) {
      // tidyでforward-backward-stepを囲む場合、forward前に存在するため保持されたままとなるパラメータを明示的に削除する必要
      prevData.dispose();
      prevMom.dispose();
      prevVar.dispose();
    }
  }

  getKeepTensors(): Tensor[] {
    return [...this.momentum.values(), ...this.variance.values()];
  }
}

export class AdamW extends Optimizer {
  momentum: Map<Parameter, Tensor>;
  variance: Map<Parameter, Tensor>;
  deviceStateSteps: number;
  constructor(
    params: Iterable<Parameter>,
    public lr = 0.001,
    public beta1 = 0.9,
    public beta2 = 0.999,
    public weightDecay = 0.01,
    public epsilon = 1e-8
  ) {
    super(params);
    this.momentum = new Map();
    this.variance = new Map();
    this.deviceStateSteps = 0;
  }

  async step(): Promise<void> {
    await super.step();
    this.deviceStateSteps++;
  }

  async stepOne(parameter: Parameter): Promise<void> {
    const T = parameter.data.getClass();
    const prevMom = this.momentum.get(parameter);
    const prevVar = this.variance.get(parameter);
    const prevData = parameter.data;
    const [momentum, variance, newData] = await tidy(async () => {
      // TODO: as anyを回避
      const grad = parameter.grad!.data as any;

      // momentum
      let m = prevMom || T.zeros(parameter.data.shape);
      m = T.add(
        T.mul(m as any, this.beta1) as any,
        T.mul(grad as any, 1 - this.beta1) as any
      );
      const mHat = T.div(
        m as any,
        1 - Math.pow(this.beta1, this.deviceStateSteps + 1)
      );

      // variance
      let v = prevVar || T.zeros(parameter.data.shape);
      v = T.add(
        T.mul(v as any, this.beta2) as any,
        T.mul(T.mul(grad as any, grad as any) as any, 1 - this.beta2) as any
      );
      const vHat = T.div(
        v as any,
        1 - Math.pow(this.beta2, this.deviceStateSteps + 1)
      );

      let newData = T.mul(prevData as any, 1 - this.lr * this.weightDecay);
      newData = T.sub(
        newData as any,
        T.mul(
          T.div(
            mHat as any,
            T.add(T.sqrt(vHat as any) as any, this.epsilon) as any
          ) as any,
          this.lr
        ) as any
      ) as any;

      return [m, v, newData];
    });

    parameter.data = newData;
    this.momentum.set(parameter, momentum);
    this.variance.set(parameter, variance);
    if (prevMom && prevVar) {
      // tidyでforward-backward-stepを囲む場合、forward前に存在するため保持されたままとなるパラメータを明示的に削除する必要
      prevData.dispose();
      prevMom.dispose();
      prevVar.dispose();
    }
  }

  getKeepTensors(): Tensor[] {
    return [...this.momentum.values(), ...this.variance.values()];
  }
}

import { defaultNNContext } from '../context';
import { CPUTensor } from '../tensor/cpuTensor';
import { Tensor } from '../tensor/tensor';
import { arange, arrayEqual } from '../util';
import {
  Add,
  BroadcastTo,
  Mul,
  NNFunction,
  Sum,
  SumTo,
  Variable,
} from './core';

export async function broadcastTo(
  x: Variable,
  shape: ReadonlyArray<number>
): Promise<Variable> {
  return await new BroadcastTo(shape).c(x);
}

export async function sumTo(
  x: Variable,
  shape: ReadonlyArray<number>
): Promise<Variable> {
  return await new SumTo(shape).c(x);
}

export async function sum(
  x: Variable,
  axis?: number | number[] | null,
  keepdims?: boolean
): Promise<Variable> {
  return await new Sum(axis, keepdims).c(x);
}

export class Sub extends NNFunction {
  async forward([lhs, rhs]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.sub(lhs as CPUTensor, rhs as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const gyShape = gy.data.shape;
    const lhsShape = this.inputs![0].data.shape;
    const rhsShape = this.inputs![1].data.shape;
    if (arrayEqual(lhsShape, rhsShape)) {
      // TODO: インスタンス共有してよいか確認
      return [gy, await neg(gy)];
    } else {
      let glhs: Variable, grhs: Variable;
      if (arrayEqual(lhsShape, gyShape)) {
        glhs = gy;
      } else {
        glhs = await new SumTo(lhsShape).c(gy);
      }
      if (arrayEqual(rhsShape, gyShape)) {
        grhs = await neg(gy);
      } else {
        grhs = await neg(await new SumTo(rhsShape).c(gy));
      }
      return [glhs, grhs];
    }
  }
}

export class Div extends NNFunction {
  async forward([lhs, rhs]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.div(lhs as CPUTensor, rhs as CPUTensor)];
  }

  // TODO: backward
}

export async function add(lhs: Variable, rhs: Variable): Promise<Variable> {
  return await new Add().c(lhs, rhs);
}

export async function sub(lhs: Variable, rhs: Variable): Promise<Variable> {
  return await new Sub().c(lhs, rhs);
}

export async function mul(lhs: Variable, rhs: Variable): Promise<Variable> {
  return await new Mul().c(lhs, rhs);
}

export async function div(lhs: Variable, rhs: Variable): Promise<Variable> {
  return await new Div().c(lhs, rhs);
}

export class Exp extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.exp(x as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const y = this.outputs?.[0]?.deref();
    if (!y) {
      throw new Error();
    }
    const gx = (await new Mul().call(y, gy))[0];
    return [gx];
  }
}

export async function exp(x: Variable): Promise<Variable> {
  return await new Exp().c(x);
}

export class Neg extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.neg(x as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const gx = await new Neg().c(gy);
    return [gx];
  }
}

export async function neg(x: Variable): Promise<Variable> {
  return await new Neg().c(x);
}

export class ReLUBackprop extends NNFunction {
  async forward([x, gx]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.reluBackprop(x as CPUTensor, gx as CPUTensor)];
  }
}

export class ReLU extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.relu(x as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new ReLUBackprop().c(this.inputs![0], gy)];
  }
}

export async function relu(x: Variable): Promise<Variable> {
  return await new ReLU().c(x);
}

export class Sigmoid extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.sigmoid(x as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: backprop可能にする
    // (1-y)*y*gyの式で単純にbackprop可能にすると効率が低下する
    // double backpropはレアなので、通常の利用の効率を優先したい
    const y = this.outputs?.[0]?.deref();
    if (!y) {
      throw new Error();
    }
    const gx = CPUTensor.sigmoidBackprop(
      y.data as CPUTensor,
      gy.data as CPUTensor
    );
    return [new Variable(gx)];
  }
}

export async function sigmoid(x: Variable): Promise<Variable> {
  return await new Sigmoid().c(x);
}

export class MatMul extends NNFunction {
  constructor(public transa = false, public transb = false) {
    super();
  }

  async forward([a, b]: Tensor[]): Promise<Tensor[]> {
    return [
      CPUTensor.gemm(a as CPUTensor, b as CPUTensor, this.transa, this.transb),
    ];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // a: [m, k], b: [k, n], y: [m, n]
    if (!this.inputs) {
      throw new Error();
    }
    const [a, b] = this.inputs;
    const ga = await new MatMul(this.transa, !this.transb).c(gy, b);
    const gb = await new MatMul(!this.transa, this.transb).c(a, gy);
    return [ga, gb];
  }
}

export async function matmul(
  a: Variable,
  b: Variable,
  transa = false,
  transb = false
): Promise<Variable> {
  return await new MatMul(transa, transb).c(a, b);
}

export class SoftmaxCrossEntropyBackward extends NNFunction {
  async forward([softmax, label, gy]: Tensor[]): Promise<Tensor[]> {
    return [
      CPUTensor.softmaxCrossEntropyBackward(
        softmax as CPUTensor,
        label as CPUTensor,
        gy as CPUTensor
      ),
    ];
  }
}

export class Softmax extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    const softmax = CPUTensor.softmax(x as CPUTensor);
    return [softmax];
  }

  // TODO: backward (学習時は、SoftmaxCrossEntropyを推奨)
}

export async function softmax(x: Variable): Promise<Variable> {
  return await new Softmax().c(x);
}

export class SoftmaxCrossEntropy extends NNFunction {
  // TODO: 中間変数の保持や開放の仕組み
  softmax?: Tensor;

  async forward([x, label]: Tensor[]): Promise<Tensor[]> {
    const softmax = CPUTensor.softmax(x as CPUTensor);
    if (defaultNNContext.get('enableBackprop')) {
      this.softmax = softmax;
    }
    const ce = CPUTensor.nllLoss(softmax, label as CPUTensor);
    return [ce];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const softmax = this.softmax;
    if (!softmax) {
      throw new Error('softmax result not stored');
    }
    const label = this.inputs![1];

    return [
      await new SoftmaxCrossEntropyBackward().c(
        new Variable(softmax),
        label,
        gy
      ),
    ];
  }
}

export async function softmaxCrossEntropy(
  x: Variable,
  label: Variable
): Promise<Variable> {
  return await new SoftmaxCrossEntropy().c(x, label);
}

export class MSELoss extends NNFunction {
  async forward([a, b]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.mseLoss(a as CPUTensor, b as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    const [a, b] = this.inputs;
    // TODO: backprop可能にする
    const [ga, gb] = CPUTensor.mseLossBackprop(
      a.data as CPUTensor,
      b.data as CPUTensor,
      gy.data as CPUTensor
    );
    return [new Variable(ga), new Variable(gb)];
  }
}

export async function mseLoss(a: Variable, b: Variable): Promise<Variable> {
  return await new MSELoss().c(a, b);
}

export class Linear extends NNFunction {
  async forward([x, weight, bias]: Tensor[]): Promise<Tensor[]> {
    let y = CPUTensor.gemm(x as CPUTensor, weight as CPUTensor, false, true);
    if (bias) {
      y = CPUTensor.add(y, bias as CPUTensor);
    }
    return [y];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    const [x, weight] = this.inputs;
    const gx = await matmul(gy, weight, false, false);
    const gweight = await matmul(gy, x, true, false);
    if (this.inputs.length === 3) {
      // grad of bias
      const b = this.inputs[2];
      const gb = await sumTo(gy, b.data.shape);
      return [gx, gweight, gb];
    } else {
      return [gx, gweight];
    }
  }
}

export async function linear(
  x: Variable,
  weight: Variable,
  bias?: Variable
): Promise<Variable> {
  if (bias) {
    return await new Linear().c(x, weight, bias);
  } else {
    return await new Linear().c(x, weight);
  }
}

export class Reshape extends NNFunction {
  xShape?: ReadonlyArray<number>;
  constructor(
    public shape: ReadonlyArray<number> | number,
    public allowZero = true
  ) {
    super();
  }
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    this.xShape = x.shape;
    const y = CPUTensor.reshape(x as CPUTensor, this.shape, this.allowZero);
    return [y];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs || !this.xShape) {
      throw new Error();
    }
    const gx = await new Reshape(this.xShape, true).c(gy);
    return [gx];
  }
}

export async function reshape(
  x: Variable,
  shape: ReadonlyArray<number> | number,
  allowZero = true
): Promise<Variable> {
  return new Reshape(shape, allowZero).c(x);
}

export class Transpose extends NNFunction {
  constructor(public axes?: ReadonlyArray<number> | null) {
    super();
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    const y = CPUTensor.transpose(x as CPUTensor, this.axes);
    return [y];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    let invertedAxes: number[];
    if (this.axes) {
      // invertedAxes = argsort(this.axes)
      const ivs = this.axes.map((v, i) => [i, v]);
      ivs.sort((a, b) => a[1] - b[1]);
      invertedAxes = ivs.map((iv) => iv[0]);
    } else {
      invertedAxes = arange(gy.data.ndim - 1, -1, -1);
    }
    const gx = await new Transpose(invertedAxes).c(gy);
    return [gx];
  }
}

export async function transpose(
  x: Variable,
  axes?: ReadonlyArray<number> | null
): Promise<Variable> {
  return new Transpose(axes).c(x);
}

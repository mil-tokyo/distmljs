import { defaultNNContext } from '../context';
import {
  max_pool2d_cpu,
  max_pool2d_with_indices_cpu,
} from '../tensor/cpu/nnfunction/max_pool2d';
import { Tensor } from '../tensor/tensor';
import { genCall } from '../tensor/tensorTypeUtil';
import {
  max_pool2d_webgl,
  max_pool2d_with_indices_webgl,
} from '../tensor/webgl/nnfunction/max_pool2d';
import { arange, arrayEqual, nonNull } from '../util';
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
    return genCall([lhs, rhs], {
      cpu: (c, [lhs, rhs]) => [c.sub(lhs, rhs)],
      webgl: (c, [lhs, rhs]) => [c.sub(lhs, rhs)],
      webgpu: (c, [lhs, rhs]) => [c.sub(lhs, rhs)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const gyShape = gy.data.shape;
    const lhsShape = nonNull(this.inputs)[0].data.shape;
    const rhsShape = nonNull(this.inputs)[1].data.shape;
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
    return genCall([lhs, rhs], {
      cpu: (c, [lhs, rhs]) => [c.div(lhs, rhs)],
      webgl: (c, [lhs, rhs]) => [c.div(lhs, rhs)],
      webgpu: (c, [lhs, rhs]) => [c.div(lhs, rhs)],
    });
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
    return genCall([x], {
      cpu: (c, [x]) => [c.exp(x)],
      webgl: (c, [x]) => [c.exp(x)],
      webgpu: (c, [x]) => [c.exp(x)],
    });
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
    return genCall([x], {
      cpu: (c, [x]) => [c.neg(x)],
      webgl: (c, [x]) => [c.neg(x)],
      webgpu: (c, [x]) => [c.neg(x)],
    });
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
    return genCall([x, gx], {
      cpu: (c, [x, gx]) => [c.reluBackprop(x, gx)],
      webgl: (c, [x, gx]) => [c.reluBackprop(x, gx)],
      webgpu: (c, [x, gx]) => [c.reluBackprop(x, gx)],
    });
  }
}

export class ReLU extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return genCall([x], {
      cpu: (c, [x]) => [c.relu(x)],
      webgl: (c, [x]) => [c.relu(x)],
      webgpu: (c, [x]) => [c.relu(x)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new ReLUBackprop().c(nonNull(this.inputs)[0], gy)];
  }
}

export async function relu(x: Variable): Promise<Variable> {
  return await new ReLU().c(x);
}

export class Sigmoid extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return genCall([x], {
      cpu: (c, [x]) => [c.sigmoid(x)],
      webgl: (c, [x]) => [c.sigmoid(x)],
      webgpu: (c, [x]) => [c.sigmoid(x)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: backprop可能にする
    // (1-y)*y*gyの式で単純にbackprop可能にすると効率が低下する
    // double backpropはレアなので、通常の利用の効率を優先したい
    const y = this.outputs?.[0]?.deref();
    if (!y) {
      throw new Error();
    }
    const [gx] = genCall([y.data, gy.data], {
      cpu: (c, [yd, gyd]) => [c.sigmoidBackprop(yd, gyd)],
      webgl: (c, [yd, gyd]) => [c.sigmoidBackprop(yd, gyd)],
      webgpu: (c, [yd, gyd]) => [c.sigmoidBackprop(yd, gyd)],
    });
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
    return genCall([a, b], {
      cpu: (c, [a, b]) => [c.gemm(a, b, this.transa, this.transb)],
      webgl: (c, [a, b]) => [c.gemm(a, b, this.transa, this.transb)],
      webgpu: (c, [a, b]) => [c.gemm(a, b, this.transa, this.transb)],
    });
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
    return genCall([softmax, label, gy], {
      cpu: (c, [softmax, label, gy]) => [
        c.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
      webgl: (c, [softmax, label, gy]) => [
        c.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
      webgpu: (c, [softmax, label, gy]) => [
        c.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
    });
  }
}

export class Softmax extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return genCall([x], {
      cpu: (c, [x]) => [c.softmax(x)],
      webgl: (c, [x]) => [c.softmax(x)],
      webgpu: (c, [x]) => [c.softmax(x)],
    });
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
    const [softmax] = genCall([x], {
      cpu: (c, [x]) => [c.softmax(x)],
      webgl: (c, [x]) => [c.softmax(x)],
      webgpu: (c, [x]) => [c.softmax(x)],
    });
    if (defaultNNContext.get('enableBackprop')) {
      this.softmax = softmax;
    }
    const ce = genCall([softmax, label], {
      cpu: (c, [softmax, label]) => [c.nllLoss(softmax, label)],
      webgl: (c, [softmax, label]) => [c.nllLoss(softmax, label)],
      webgpu: (c, [softmax, label]) => [c.nllLoss(softmax, label)],
    });
    return ce;
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const softmax = this.softmax;
    if (!softmax) {
      throw new Error('softmax result not stored');
    }
    const label = nonNull(this.inputs)[1];

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
    return genCall([a, b], {
      cpu: (c, [a, b]) => [c.mseLoss(a, b)],
      webgl: (c, [a, b]) => [c.mseLoss(a, b)],
      webgpu: (c, [a, b]) => [c.mseLoss(a, b)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    const [a, b] = this.inputs;
    // TODO: backprop可能にする
    const [ga, gb] = genCall([a.data, b.data, gy.data], {
      cpu: (c, [ad, bd, gyd]) => c.mseLossBackprop(ad, bd, gyd),
      webgl: (c, [ad, bd, gyd]) => c.mseLossBackprop(ad, bd, gyd),
      webgpu: (c, [ad, bd, gyd]) => c.mseLossBackprop(ad, bd, gyd),
    });
    return [new Variable(ga), new Variable(gb)];
  }
}

export async function mseLoss(a: Variable, b: Variable): Promise<Variable> {
  return await new MSELoss().c(a, b);
}

export class Linear extends NNFunction {
  async forward([x, weight, bias]: Tensor[]): Promise<Tensor[]> {
    let [y] = genCall([x, weight], {
      cpu: (c, [x, weight]) => [c.gemm(x, weight, false, true)],
      webgl: (c, [x, weight]) => [c.gemm(x, weight, false, true)],
      webgpu: (c, [x, weight]) => [c.gemm(x, weight, false, true)],
    });
    if (bias) {
      [y] = genCall([y, bias], {
        cpu: (c, [y, bias]) => [c.add(y, bias)],
        webgl: (c, [y, bias]) => [c.add(y, bias)],
        webgpu: (c, [y, bias]) => [c.add(y, bias)],
      });
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
    return genCall([x], {
      cpu: (c, [x]) => [c.reshape(x, this.shape, this.allowZero)],
      webgl: (c, [x]) => [c.reshape(x, this.shape, this.allowZero)],
      webgpu: (c, [x]) => [c.reshape(x, this.shape, this.allowZero)],
    });
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
    return genCall([x], {
      cpu: (c, [x]) => [c.transpose(x, this.axes)],
      webgl: (c, [x]) => [c.transpose(x, this.axes)],
      webgpu: (c, [x]) => [c.transpose(x, this.axes)],
    });
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

export class Flatten extends NNFunction {
  xShape?: ReadonlyArray<number>;
  constructor() {
    super();
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    this.xShape = x.shape;
    const batch = x.shape[0] || 1;
    return genCall([x], {
      cpu: (c, [x]) => [c.reshape(x, [batch, -1])],
      webgl: (c, [x]) => [c.reshape(x, [batch, -1])],
      webgpu: (c, [x]) => [c.reshape(x, [batch, -1])],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs || !this.xShape) {
      throw new Error();
    }
    const gx = await new Reshape(this.xShape, true).c(gy);
    return [gx];
  }
}

/**
 * Flattens variable into 2D (batch, -1). Note: CPUTensor.flatten flattens into 1D.
 * @param x
 * @returns
 */
export async function flatten(x: Variable): Promise<Variable> {
  return new Flatten().c(x);
}

export interface MaxPool2dParamsReturnIndicesFalse {
  kernelSize: number;
  stride?: number;
  padding?: number;
  dilation?: number;
  returnIndices?: false;
  ceilMode?: boolean;
}

export interface MaxPool2dParamsReturnIndicesTrue {
  kernelSize: number;
  stride?: number;
  padding?: number;
  dilation?: number;
  returnIndices: true;
  ceilMode?: boolean;
}

export type MaxPool2dParams =
  | MaxPool2dParamsReturnIndicesTrue
  | MaxPool2dParamsReturnIndicesFalse;

export class MaxPool2d extends NNFunction {
  kernelSize: number; // TODO: support [number, number] to specify different size for height and width
  stride: number;
  padding: number;
  dilation: number;
  returnIndices: boolean;
  ceilMode: boolean;

  constructor(params: MaxPool2dParams) {
    super();
    const {
      kernelSize,
      stride,
      padding = 0,
      dilation = 1,
      returnIndices = false,
      ceilMode = false,
    } = params;
    this.kernelSize = kernelSize;
    this.stride = stride || kernelSize;
    this.padding = padding;
    this.dilation = dilation;
    this.returnIndices = returnIndices;
    this.ceilMode = ceilMode;
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    if (this.returnIndices) {
      return genCall([x], {
        cpu: (c, [x]) =>
          max_pool2d_with_indices_cpu(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: true,
            ceilMode: this.ceilMode,
          }),
        webgl: (c, [x]) =>
          max_pool2d_with_indices_webgl(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: true,
            ceilMode: this.ceilMode,
          }),
      });
    } else {
      return genCall([x], {
        cpu: (c, [x]) => [
          max_pool2d_cpu(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: false,
            ceilMode: this.ceilMode,
          }),
        ],
        webgl: (c, [x]) => [
          max_pool2d_webgl(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: false,
            ceilMode: this.ceilMode,
          }),
        ],
      });
    }
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    throw new Error('not implemented');
  }
}

export async function max_pool2d(
  x: Variable,
  params: MaxPool2dParamsReturnIndicesFalse
): Promise<Variable> {
  return new MaxPool2d(params).c(x);
}

export async function max_pool2d_with_indices(
  x: Variable,
  params: MaxPool2dParamsReturnIndicesTrue
): Promise<Variable[]> {
  return new MaxPool2d(params).call(x);
}

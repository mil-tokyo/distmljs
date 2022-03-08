import { defaultNNContext } from '../context';
import {
  avg_pool2d_backprop_cpu,
  avg_pool2d_cpu,
} from '../tensor/cpu/nnfunction/avg_pool2d';
import {
  conv2d_backprop_gb_cpu,
  conv2d_backprop_gxgw_cpu,
  conv2d_cpu,
} from '../tensor/cpu/nnfunction/conv2d';
import {
  max_pool2d_backprop_cpu,
  max_pool2d_cpu,
  max_pool2d_with_indices_cpu,
} from '../tensor/cpu/nnfunction/max_pool2d';
import * as cpuCore from '../tensor/cpu/core';
import * as webglCore from '../tensor/webgl/core';
import * as webgpuCore from '../tensor/webgpu/core';
import { Tensor } from '../tensor/tensor';
import {
  genCall,
  isAllCPUTensor,
  isAllWebGLTensor,
} from '../tensor/tensorTypeUtil';
import {
  avg_pool2d_backprop_webgl,
  avg_pool2d_webgl,
} from '../tensor/webgl/nnfunction/avg_pool2d';
import {
  conv2d_backprop_gb_webgl,
  conv2d_backprop_gxgw_webgl,
  conv2d_webgl,
} from '../tensor/webgl/nnfunction/conv2d';
import {
  max_pool2d_backprop_webgl,
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
import {
  batch_norm_backprop_cpu,
  batch_norm_cpu,
  layer_norm_backprop_cpu,
  layer_norm_cpu,
} from '../tensor/cpu/nnfunction/batch_norm';
import {
  batch_norm_backprop_webgl,
  batch_norm_webgl,
} from '../tensor/webgl/nnfunction/batch_norm';
import { CPUTensor } from '../tensor';
import {
  embedding_backprop_cpu,
  embedding_cpu,
} from '../tensor/cpu/nnfunction/embedding';

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
      all: (c, [lhs, rhs]) => [c.sub(lhs, rhs)],
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
      all: (c, [lhs, rhs]) => [c.div(lhs, rhs)],
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
      all: (c, [x]) => [c.exp(x)],
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
      all: (c, [x]) => [c.neg(x)],
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
      cpu: (c, [x, gx]) => [cpuCore.reluBackprop(x, gx)],
      webgl: (c, [x, gx]) => [webglCore.reluBackprop(x, gx)],
      webgpu: (c, [x, gx]) => [webgpuCore.reluBackprop(x, gx)],
    });
  }
}

export class ReLU extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return genCall([x], {
      all: (c, [x]) => [c.relu(x)],
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
      all: (c, [x]) => [c.sigmoid(x)],
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
      cpu: (c, [yd, gyd]) => [cpuCore.sigmoidBackprop(yd, gyd)],
      webgl: (c, [yd, gyd]) => [webglCore.sigmoidBackprop(yd, gyd)],
      webgpu: (c, [yd, gyd]) => [webgpuCore.sigmoidBackprop(yd, gyd)],
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
      all: (c, [a, b]) => [c.gemm(a, b, this.transa, this.transb)],
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

export class SoftmaxBackward extends NNFunction {
  async forward([softmax, gy]: Tensor[]): Promise<Tensor[]> {
    return genCall([softmax, gy], {
      cpu: (c, [softmax, gy]) => [cpuCore.softmaxBackward(softmax, gy)],
    });
  }
}

export class SoftmaxCrossEntropyBackward extends NNFunction {
  async forward([softmax, label, gy]: Tensor[]): Promise<Tensor[]> {
    return genCall([softmax, label, gy], {
      cpu: (c, [softmax, label, gy]) => [
        cpuCore.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
      webgl: (c, [softmax, label, gy]) => [
        webglCore.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
      webgpu: (c, [softmax, label, gy]) => [
        webgpuCore.softmaxCrossEntropyBackward(softmax, label, gy),
      ],
    });
  }
}

export class Softmax extends NNFunction {
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return genCall([x], {
      cpu: (c, [x]) => [cpuCore.softmax(x)],
      webgl: (c, [x]) => [webglCore.softmax(x)],
      webgpu: (c, [x]) => [webgpuCore.softmax(x)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const softmax = this.outputs?.[0]?.deref();
    if (!softmax) {
      throw new Error();
    }
    return [await new SoftmaxBackward().c(softmax, gy)];
  }
}

export async function softmax(x: Variable): Promise<Variable> {
  return await new Softmax().c(x);
}

export class SoftmaxCrossEntropy extends NNFunction {
  // TODO: 中間変数の保持や開放の仕組み
  softmax?: Tensor;

  async forward([x, label]: Tensor[]): Promise<Tensor[]> {
    const [softmax] = genCall([x], {
      cpu: (c, [x]) => [cpuCore.softmax(x)],
      webgl: (c, [x]) => [webglCore.softmax(x)],
      webgpu: (c, [x]) => [webgpuCore.softmax(x)],
    });
    if (defaultNNContext.get('enableBackprop')) {
      this.softmax = softmax;
    }
    const ce = genCall([softmax, label], {
      cpu: (c, [softmax, label]) => [cpuCore.nllLoss(softmax, label)],
      webgl: (c, [softmax, label]) => [webglCore.nllLoss(softmax, label)],
      webgpu: (c, [softmax, label]) => [webgpuCore.nllLoss(softmax, label)],
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
      cpu: (c, [a, b]) => [cpuCore.mseLoss(a, b)],
      webgl: (c, [a, b]) => [webglCore.mseLoss(a, b)],
      webgpu: (c, [a, b]) => [webgpuCore.mseLoss(a, b)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    const [a, b] = this.inputs;
    // TODO: backprop可能にする
    const [ga, gb] = genCall([a.data, b.data, gy.data], {
      cpu: (c, [ad, bd, gyd]) => cpuCore.mseLossBackprop(ad, bd, gyd),
      webgl: (c, [ad, bd, gyd]) => webglCore.mseLossBackprop(ad, bd, gyd),
      webgpu: (c, [ad, bd, gyd]) => webgpuCore.mseLossBackprop(ad, bd, gyd),
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
      all: (c, [x, weight]) => [c.gemm(x, weight, false, true)],
    });
    if (bias) {
      [y] = genCall([y, bias], {
        all: (c, [y, bias]) => [c.add(y, bias)],
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
      all: (c, [x]) => [c.reshape(x, this.shape, this.allowZero)],
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
      all: (c, [x]) => [c.transpose(x, this.axes)],
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
      all: (c, [x]) => [c.reshape(x, [batch, -1])],
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
  // indexのモード。true='spatial'。spatialの場合は2D平面内でのインデックス(PyTorchと同様)、flattenの場合は4Dテンソル全体におけるインデックスが返る(ONNXと同様)。
  returnIndices: true | 'spatial' | 'flatten';
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
  returnIndices: boolean | 'spatial' | 'flatten';
  ceilMode: boolean;
  idx?: Tensor;
  xShape?: ReadonlyArray<number>;

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
    if (this.returnIndices || defaultNNContext.get('enableBackprop')) {
      const rit = this.returnIndices || true;
      const [max, idx] = genCall([x], {
        cpu: (c, [x]) =>
          max_pool2d_with_indices_cpu(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: rit,
            ceilMode: this.ceilMode,
          }),
        webgl: (c, [x]) =>
          max_pool2d_with_indices_webgl(x, {
            kernelSize: this.kernelSize,
            stride: this.stride,
            padding: this.padding,
            dilation: this.dilation,
            returnIndices: rit,
            ceilMode: this.ceilMode,
          }),
      });
      if (defaultNNContext.get('enableBackprop')) {
        this.xShape = x.shape;
        this.idx = idx;
      }
      if (this.returnIndices) {
        return [max, idx];
      } else {
        return [max];
      }
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
    // TODO: backprop可能にする
    const [gxd] = genCall([nonNull(this.idx), gy.data], {
      cpu: (c, [idx, gyd]) => [
        max_pool2d_backprop_cpu(idx, gyd, nonNull(this.xShape), {
          kernelSize: this.kernelSize,
          stride: this.stride,
          padding: this.padding,
          dilation: this.dilation,
          ceilMode: this.ceilMode,
          returnIndices: this.returnIndices || true,
        }),
      ],
      webgl: (c, [idx, gyd]) => [
        max_pool2d_backprop_webgl(idx, gyd, nonNull(this.xShape), {
          kernelSize: this.kernelSize,
          stride: this.stride,
          padding: this.padding,
          dilation: this.dilation,
          ceilMode: this.ceilMode,
          returnIndices: this.returnIndices || true,
        }),
      ],
    });
    return [new Variable(gxd)];
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

export class AdaptiveMaxPool2d extends NNFunction {
  outputSize: number[];
  returnIndices: boolean | 'spatial' | 'flatten';
  idx?: Tensor;
  xShape?: ReadonlyArray<number>;

  constructor(
    outputSize: number | number[],
    returnIndices: boolean | 'spatial' | 'flatten' = false
  ) {
    super();
    if (typeof outputSize === 'number') {
      this.outputSize = [outputSize, outputSize];
    } else {
      this.outputSize = outputSize;
    }
    if (
      this.outputSize.length !== 2 ||
      this.outputSize[0] !== 1 ||
      this.outputSize[1] !== 1
    ) {
      throw new Error(
        'AdaptiveMaxPool2d: currently supports only outputSize===1.'
      );
      // TODO: support other size (needs check PyTorch's implementation of size computation)
    }
    this.returnIndices = returnIndices;
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    const kernelSize = [x.shape[2], x.shape[3]]; // for 1x1 output
    if (this.returnIndices || defaultNNContext.get('enableBackprop')) {
      const rit = this.returnIndices || true;
      const params = {
        kernelSize,
        stride: kernelSize,
        padding: 0,
        dilation: 1,
        returnIndices: rit,
        ceilMode: false,
      };
      const [max, idx] = genCall([x], {
        cpu: (c, [x]) => max_pool2d_with_indices_cpu(x, params),
        webgl: (c, [x]) => max_pool2d_with_indices_webgl(x, params),
      });
      if (defaultNNContext.get('enableBackprop')) {
        this.xShape = x.shape;
        this.idx = idx;
      }
      if (this.returnIndices) {
        return [max, idx];
      } else {
        return [max];
      }
    } else {
      const params = {
        kernelSize,
        stride: kernelSize,
        padding: 0,
        dilation: 1,
        returnIndices: false as const,
        ceilMode: false,
      };
      return genCall([x], {
        cpu: (c, [x]) => [max_pool2d_cpu(x, params)],
        webgl: (c, [x]) => [max_pool2d_webgl(x, params)],
      });
    }
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: backprop可能にする
    const xShape = nonNull(this.xShape);
    const kernelSize = [xShape[2], xShape[3]]; // for 1x1 output
    const params = {
      kernelSize,
      stride: kernelSize,
      padding: 0,
      dilation: 1,
      returnIndices: this.returnIndices || true,
      ceilMode: false,
    };
    const [gxd] = genCall([nonNull(this.idx), gy.data], {
      cpu: (c, [idx, gyd]) => [
        max_pool2d_backprop_cpu(idx, gyd, xShape, params),
      ],
      webgl: (c, [idx, gyd]) => [
        max_pool2d_backprop_webgl(idx, gyd, xShape, params),
      ],
    });
    return [new Variable(gxd)];
  }
}

export async function adaptive_max_pool2d(
  x: Variable,
  outputSize: number | number[],
  returnIndices: boolean | 'spatial' | 'flatten' = false
): Promise<Variable> {
  return new AdaptiveMaxPool2d(outputSize, returnIndices).c(x);
}

export interface AvgPool2dParams {
  kernelSize: number | number[];
  stride?: number | number[];
  padding?: number | number[];
  ceilMode?: boolean;
  countIncludePad?: boolean;
  divisorOverride?: number;
}

export class AvgPool2d extends NNFunction {
  kernelSize: number | number[];
  stride: number | number[];
  padding: number | number[];
  ceilMode: boolean;
  countIncludePad: boolean;
  divisorOverride?: number;
  xShape?: ReadonlyArray<number>;

  constructor(params: AvgPool2dParams) {
    super();
    const {
      kernelSize,
      stride,
      padding = 0,
      ceilMode = false,
      countIncludePad = true,
      divisorOverride = undefined,
    } = params;
    this.kernelSize = kernelSize;
    this.stride = stride || kernelSize;
    this.padding = padding;
    this.ceilMode = ceilMode;
    this.countIncludePad = countIncludePad;
    this.divisorOverride = divisorOverride;
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    const params = {
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
      ceilMode: this.ceilMode,
      countIncludePad: this.countIncludePad,
      divisorOverride: this.divisorOverride,
    };
    this.xShape = x.shape;
    return genCall([x], {
      cpu: (c, [x]) => [avg_pool2d_cpu(x, params)],
      webgl: (c, [x]) => [avg_pool2d_webgl(x, params)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: backprop可能にする
    const params = {
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
      ceilMode: this.ceilMode,
      countIncludePad: this.countIncludePad,
      divisorOverride: this.divisorOverride,
    };
    const [gxd] = genCall([gy.data], {
      cpu: (c, [gyd]) => [
        avg_pool2d_backprop_cpu(gyd, nonNull(this.xShape), params),
      ],
      webgl: (c, [gyd]) => [
        avg_pool2d_backprop_webgl(gyd, nonNull(this.xShape), params),
      ],
    });
    return [new Variable(gxd)];
  }
}

export async function avg_pool2d(
  x: Variable,
  params: AvgPool2dParams
): Promise<Variable> {
  return new AvgPool2d(params).c(x);
}

export class AdaptiveAvgPool2d extends NNFunction {
  outputSize: number[];
  xShape?: ReadonlyArray<number>;

  constructor(outputSize: number | number[]) {
    super();
    if (typeof outputSize === 'number') {
      this.outputSize = [outputSize, outputSize];
    } else {
      this.outputSize = outputSize;
    }
    if (
      this.outputSize.length !== 2 ||
      this.outputSize[0] !== 1 ||
      this.outputSize[1] !== 1
    ) {
      throw new Error(
        'AdaptiveMaxPool2d: currently supports only outputSize===1.'
      );
      // TODO: support other size (needs check PyTorch's implementation of size computation)
    }
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    const kernelSize = [x.shape[2], x.shape[3]]; // for 1x1 output
    const params = {
      kernelSize,
      stride: kernelSize,
      padding: 0,
      countIncludePad: true,
      ceilMode: false,
    };
    this.xShape = x.shape;
    return genCall([x], {
      cpu: (c, [x]) => [avg_pool2d_cpu(x, params)],
      webgl: (c, [x]) => [avg_pool2d_webgl(x, params)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: backprop可能にする
    const xShape = nonNull(this.xShape);
    const kernelSize = [xShape[2], xShape[3]]; // for 1x1 output
    const params = {
      kernelSize,
      stride: kernelSize,
      padding: 0,
      countIncludePad: true,
      ceilMode: false,
    };
    const [gxd] = genCall([gy.data], {
      cpu: (c, [gyd]) => [avg_pool2d_backprop_cpu(gyd, xShape, params)],
      webgl: (c, [gyd]) => [avg_pool2d_backprop_webgl(gyd, xShape, params)],
    });
    return [new Variable(gxd)];
  }
}

export async function adaptive_avg_pool2d(
  x: Variable,
  outputSize: number | number[]
): Promise<Variable> {
  return new AdaptiveAvgPool2d(outputSize).c(x);
}

export interface Conv2dParams {
  stride?: number | [number, number];
  padding?: number | [number, number] | [number, number, number, number];
  dilation?: number | [number, number];
  groups?: number;
}

export class Conv2d extends NNFunction {
  stride: number | [number, number];
  padding: number | [number, number] | [number, number, number, number]; //TODO: support 'same' and 'valid'
  dilation: number | [number, number];
  groups: number;
  xShape?: ReadonlyArray<number>;
  wShape?: ReadonlyArray<number>;
  x?: Tensor;
  weight?: Tensor;
  hasBias?: boolean;

  constructor(params: Conv2dParams) {
    super();
    const { stride = 1, padding = 0, dilation = 1, groups = 1 } = params;
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;
  }

  async forward([x, weight, bias]: Tensor[]): Promise<Tensor[]> {
    if (defaultNNContext.get('enableBackprop')) {
      this.xShape = x.shape;
      this.wShape = weight.shape;
      this.x = x;
      this.weight = weight;
      this.hasBias = !!bias;
    }
    const params = {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
    };

    if (bias) {
      return genCall([x, weight, bias], {
        cpu: (c, [x, weight, bias]) => [conv2d_cpu(x, weight, bias, params)],
        webgl: (c, [x, weight, bias]) => [
          conv2d_webgl(x, weight, bias, params),
        ],
      });
    } else {
      return genCall([x, weight], {
        cpu: (c, [x, weight]) => [conv2d_cpu(x, weight, undefined, params)],
        webgl: (c, [x, weight]) => [conv2d_webgl(x, weight, undefined, params)],
      });
    }
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    // TODO: convtransposeを実装し、conv2d_backprop_gx_cpuの実装を共用するとともにbackprop可能にする
    const params = {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
    };
    const [gxd, gwd] = genCall(
      [gy.data, nonNull(this.x), nonNull(this.weight)],
      {
        cpu: (c, [gyd, x, weight]) =>
          conv2d_backprop_gxgw_cpu(gyd, x, weight, false, false, params),

        webgl: (c, [gyd, x, weight]) =>
          conv2d_backprop_gxgw_webgl(gyd, x, weight, false, false, params),
      }
    );
    if (this.hasBias) {
      const [gbd] = genCall([gy.data], {
        cpu: (c, [gyd]) => [conv2d_backprop_gb_cpu(gyd)],
        webgl: (c, [gyd]) => [conv2d_backprop_gb_webgl(gyd)],
      });
      return [new Variable(gxd), new Variable(gwd), new Variable(gbd)];
    } else {
      return [new Variable(gxd), new Variable(gwd)];
    }
  }
}

export function conv2d(
  x: Variable,
  weight: Variable,
  params?: Conv2dParams
): Promise<Variable>;
export function conv2d(
  x: Variable,
  weight: Variable,
  bias: Variable | undefined,
  params?: Conv2dParams
): Promise<Variable>;
export async function conv2d(
  x: Variable,
  weight: Variable,
  biasOrParams?: Conv2dParams | Variable,
  params?: Conv2dParams
): Promise<Variable> {
  if (biasOrParams instanceof Variable) {
    return new Conv2d(params || {}).c(x, weight, biasOrParams);
  } else if (biasOrParams) {
    return new Conv2d(biasOrParams || {}).c(x, weight);
  } else {
    // x, weight, undefined, params
    return new Conv2d(params || {}).c(x, weight);
  }
}

export interface BatchNormParams {
  numFeatures: number;
  training: boolean;
  eps?: number;
  momentum?: number;
  trackRunningStats: boolean;
}

export class BatchNormFunction extends NNFunction {
  numFeatures: number;
  training: boolean;
  eps: number;
  momentum?: number;
  trackRunningStats: boolean;
  statsForBackprop?: Tensor;

  constructor(params: BatchNormParams) {
    super();
    const {
      numFeatures,
      training,
      trackRunningStats,
      eps = 1e-5,
      momentum = 0.1,
    } = params;
    this.numFeatures = numFeatures;
    this.training = training;
    this.eps = eps;
    this.momentum = momentum;
    this.trackRunningStats = trackRunningStats;
  }

  async forward([
    x,
    weight,
    bias,
    runningMean,
    runningVar,
    numBatchesTracked,
  ]: Tensor[]): Promise<Tensor[]> {
    const params = {
      axis: 1,
      training: this.training,
      eps: this.eps,
      momentum: this.momentum,
      trackRunningStats: this.trackRunningStats,
    };

    if (!(weight && bias)) {
      // TODO: 場合分け
      throw new Error(
        'BatchNormND without weight, bias is not yet implemented'
      );
    }
    let outputs: {
      y: Tensor;
      statsForBackprop: Tensor;
      updatedRunningStats: {
        runningMean: Tensor;
        runningVar: Tensor;
        numBatchesTracked: Tensor;
      } | null;
    };
    if (runningMean && runningVar && numBatchesTracked) {
      const ts = [x, weight, bias, runningMean, runningVar, numBatchesTracked];
      if (isAllCPUTensor(ts)) {
        outputs = batch_norm_cpu(
          ts[0],
          { weight: ts[1], bias: ts[2] },
          { runningMean: ts[3], runningVar: ts[4], numBatchesTracked: ts[5] },
          params
        );
      } else if (isAllWebGLTensor(ts)) {
        outputs = batch_norm_webgl(
          ts[0],
          { weight: ts[1], bias: ts[2] },
          { runningMean: ts[3], runningVar: ts[4], numBatchesTracked: ts[5] },
          params
        );
      } else {
        throw new Error('not implemented');
      }
    } else {
      const ts = [x, weight, bias];
      if (isAllCPUTensor(ts)) {
        outputs = batch_norm_cpu(
          ts[0],
          { weight: ts[1], bias: ts[2] },
          null,
          params
        );
      } else if (isAllWebGLTensor(ts)) {
        outputs = batch_norm_webgl(
          ts[0],
          { weight: ts[1], bias: ts[2] },
          null,
          params
        );
      } else {
        throw new Error('not implemented');
      }
    }

    if (defaultNNContext.get('enableBackprop')) {
      this.statsForBackprop = outputs.statsForBackprop;
    }

    if (outputs.updatedRunningStats) {
      return [
        outputs.y,
        outputs.updatedRunningStats.runningMean,
        outputs.updatedRunningStats.runningVar,
        outputs.updatedRunningStats.numBatchesTracked,
      ];
    } else {
      return [outputs.y];
    }
  }

  async backward([gy]: Variable[]): Promise<(Variable | null)[]> {
    const [gxd, gwd, gbd] = genCall(
      [nonNull(this.inputs)[0].data, gy.data, nonNull(this.statsForBackprop)],
      {
        cpu: (c, [x, gyd, sfb]) => {
          const xwb = batch_norm_backprop_cpu(x, gyd, sfb, 1);
          return [xwb.gx, xwb.gweight, xwb.gbias];
        },
        webgl: (c, [x, gyd, sfb]) => {
          const xwb = batch_norm_backprop_webgl(x, gyd, sfb, 1);
          return [xwb.gx, xwb.gweight, xwb.gbias];
        },
      }
    );
    return [
      new Variable(gxd),
      new Variable(gwd),
      new Variable(gbd),
      null,
      null,
      null,
    ];
  }
}

export interface LayerNormParams {
  normalizedShape: ReadonlyArray<number>;
  eps?: number;
}

export class LayerNormFunction extends NNFunction {
  normalizedShape: ReadonlyArray<number>;
  eps: number;
  statsForBackprop?: Tensor;

  constructor(params: LayerNormParams) {
    super();
    const { normalizedShape, eps = 1e-5 } = params;
    this.eps = eps;
    this.normalizedShape = normalizedShape;
  }

  async forward([x, weight, bias]: Tensor[]): Promise<Tensor[]> {
    const params = {
      normalizedShape: this.normalizedShape,
      eps: this.eps,
    };

    if (!(weight && bias)) {
      // TODO: 場合分け
      throw new Error('LayerNorm without weight, bias is not yet implemented');
    }
    let outputs: {
      y: Tensor;
      statsForBackprop: Tensor;
    };
    const ts = [x, weight, bias];
    if (isAllCPUTensor(ts)) {
      outputs = layer_norm_cpu(ts[0], { weight: ts[1], bias: ts[2] }, params);
    } else {
      throw new Error('not implemented');
    }

    if (defaultNNContext.get('enableBackprop')) {
      this.statsForBackprop = outputs.statsForBackprop;
    }

    return [outputs.y];
  }

  async backward([gy]: Variable[]): Promise<(Variable | null)[]> {
    const params = {
      normalizedShape: this.normalizedShape,
      eps: this.eps,
    };
    const [gxd, gwd, gbd] = genCall(
      [
        nonNull(this.inputs)[0].data,
        nonNull(this.inputs)[1].data,
        gy.data,
        nonNull(this.statsForBackprop),
      ],
      {
        cpu: (c, [x, w, gyd, sfb]) => {
          const xwb = layer_norm_backprop_cpu(x, w, gyd, sfb, params);
          return [xwb.gx, xwb.gweight, xwb.gbias];
        },
      }
    );
    return [new Variable(gxd), new Variable(gwd), new Variable(gbd)];
  }
}

export class EmbeddingFunction extends NNFunction {
  constructor(
    public readonly numEmbeddings: number,
    public readonly embeddingDim: number
  ) {
    super();
  }

  async forward([x, weight]: Tensor[]): Promise<Tensor[]> {
    const ts = [x, weight];
    let output: CPUTensor;
    if (isAllCPUTensor(ts)) {
      output = embedding_cpu(ts[0], ts[1]);
    } else {
      throw new Error('not implemented');
    }

    return [output];
  }

  async backward([gy]: Variable[]): Promise<(Variable | null)[]> {
    const [gwd] = genCall([nonNull(this.inputs)[0].data, gy.data], {
      cpu: (c, [x, gyd]) => {
        return [
          embedding_backprop_cpu(x, gyd, this.numEmbeddings, this.embeddingDim),
        ];
      },
    });
    return [null, new Variable(gwd)];
  }
}

import { CPUTensor } from './cpu/cpuTensor';
import { Backend } from '../backend';
import { DType } from '../dtype';
import { WebGLTensor } from './webgl/webglTensor';
import { WebGPUTensor } from './webgpu/webgpuTensor';

export abstract class Tensor {
  readonly backend: Backend;
  readonly dtype: DType;
  readonly shape: ReadonlyArray<number>;
  readonly strides: ReadonlyArray<number>;
  readonly ndim: number;
  readonly size: number;

  constructor(backend: Backend, shape: ArrayLike<number>, dtype: DType) {
    this.backend = backend;
    this.dtype = dtype;
    this.shape = Array.from(shape).map((v) => Number(v) | 0);
    this.ndim = this.shape.length;
    const strides: number[] = [];
    let size = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      strides.unshift(size);
      size *= this.shape[i];
    }
    this.strides = strides;
    this.size = size;
  }

  abstract alias(shape?: ArrayLike<number>): Tensor;
  abstract to(backend: 'cpu'): Promise<CPUTensor>;
  abstract to(backend: Backend): Promise<Tensor>;
  abstract getClass():
    | TensorStatic<CPUTensor>
    | TensorStatic<WebGLTensor>
    | TensorStatic<WebGPUTensor>;
  abstract toArrayAsync(): Promise<number[]>;
  abstract copy(): Tensor;
  abstract reshape(
    shape: ReadonlyArray<number> | number,
    allowZero: boolean
  ): Tensor;
  abstract transpose(axes?: ReadonlyArray<number> | null): Tensor;

  abstract dispose(): void;
}

// Static methods implemented by all backeds
export interface TensorStatic<B extends Tensor> {
  s: (value: number) => B;
  zeros: (shape: ReadonlyArray<number>, dtype?: DType) => B;
  ones: (shape: ReadonlyArray<number>, dtype?: DType) => B;
  fromArray: (
    data: ArrayLike<number>,
    shape?: ArrayLike<number>,
    dtype?: DType
  ) => B;
  add: (lhs: B, rhs: B) => B;
  sub: (lhs: B, rhs: B) => B;
  mul: (lhs: B, rhs: B) => B;
  div: (lhs: B, rhs: B) => B;
  pow: (lhs: B, rhs: B) => B;
  abs: (x: B) => B;
  acos: (x: B) => B;
  acosh: (x: B) => B;
  asin: (x: B) => B;
  asinh: (x: B) => B;
  atan: (x: B) => B;
  atanh: (x: B) => B;
  cos: (x: B) => B;
  cosh: (x: B) => B;
  exp: (x: B) => B;
  log: (x: B) => B;
  neg: (x: B) => B;
  relu: (x: B) => B;
  sigmoid: (x: B) => B;
  sin: (x: B) => B;
  sinh: (x: B) => B;
  sqrt: (x: B) => B;
  square: (x: B) => B;
  tan: (x: B) => B;
  tanh: (x: B) => B;
  gemm: (a: B, b: B, transa?: boolean, transb?: boolean) => B;
  dot: (a: B, b: B) => B;
  broadcastTo: (x: B, shape: ReadonlyArray<number>) => B;
  sum: (x: B, axis?: number | number[] | null, keepdims?: boolean) => B;
  sumTo: (x: B, shape: ReadonlyArray<number>) => B;
  reshape: (
    x: B,
    shape: ReadonlyArray<number> | number,
    allowZero?: boolean
  ) => B;
  transpose: (x: B, axes?: ReadonlyArray<number> | null) => B;
  flatten: (x: B) => B;
  ravel: (x: B) => B;
  squeeze: (input: B, dim?: number) => B;
  unsqueeze: (input: B, dim: number) => B;
  full: (shape: ArrayLike<number>, fillValue: number, dtype?: DType) => B;
}

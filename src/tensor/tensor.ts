import { CPUTensor } from './cpu/cpuTensor';
import { Backend } from '../backend';
import { DType } from '../dtype';
import { WebGLTensor } from './webgl/webglTensor';
import { WebGPUTensor } from './webgpu/webgpuTensor';

/**
 * Base class for tensor.
 *
 * Note: Tensor does not hold backpropagation information.
 */
export abstract class Tensor {
  /**
   * The backend the tensor is on.
   */
  readonly backend: Backend;
  /**
   * Type of element.
   */
  readonly dtype: DType;
  /**
   * Shape of tensor.
   */
  readonly shape: ReadonlyArray<number>;
  /**
   * Stride of each dimension. Unit is element (not byte).
   */
  readonly strides: ReadonlyArray<number>;
  /**
   * The number of dimensions. ndim === shape.length
   */
  readonly ndim: number;
  /**
   * The total number of elements in the tensor.
   */
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

  /**
   * Make alias of tensor, which shares buffer.
   * @param shape shape of returned tensor.
   */
  abstract alias(shape?: ArrayLike<number>): Tensor;
  /**
   * Copy tensor to another backend.
   * @param backend the backend the returned tensor is located. If backend is same as given tensor, alias is returned.
   */
  abstract to(backend: 'cpu'): Promise<CPUTensor>;
  abstract to(backend: Backend): Promise<Tensor>;
  abstract getClass():
    | TensorStatic<CPUTensor>
    | TensorStatic<WebGLTensor>
    | TensorStatic<WebGPUTensor>;
  abstract toArrayAsync(): Promise<number[]>;
  /**
   * Copy tensor.
   */
  abstract copy(): Tensor;
  /**
   * Return reshaped tensor, which shares buffer with original tensor.
   * @param shape shape of returned tensor.
   * @param allowZero when true,
   */
  abstract reshape(
    shape: ReadonlyArray<number> | number,
    allowZero: boolean
  ): Tensor;
  /**
   * Return reshaped tensor, which shares buffer with original tensor.
   * @param axes order of axes. If omitted, reverses axes.
   * @example
   * ```
   * const x = CPUTensor.ones([2, 3])
   * x.transpose().shape // [3, 2]
   * ```
   * @example
   * ```
   * const x = CPUTensor.ones([2, 3, 4])
   * x.transpose([1, 2, 0]).shape // [3, 4, 2]
   * ```
   */
  abstract transpose(axes?: ReadonlyArray<number> | null): Tensor;

  /**
   * Disposes buffer. For WebGLTensor / WebGPUTensor, the buffer will not be garbage collected, so calling dispose() or using kakiage.tidy() is needed.
   */
  abstract dispose(): void;
}

/**
 * Static methods implemented by all backeds
 */
export interface TensorStatic<B extends Tensor> {
  s: (value: number) => B;
  zeros: (shape: ReadonlyArray<number>, dtype?: DType) => B;
  ones: (shape: ReadonlyArray<number>, dtype?: DType) => B;
  fromArray: (
    data: ArrayLike<number>,
    shape?: ArrayLike<number>,
    dtype?: DType
  ) => B;
  add: (lhs: B | number, rhs: B | number) => B;
  sub: (lhs: B | number, rhs: B | number) => B;
  mul: (lhs: B | number, rhs: B | number) => B;
  div: (lhs: B | number, rhs: B | number) => B;
  pow: (lhs: B | number, rhs: B | number) => B;
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
  cat(tensors: ReadonlyArray<B>, axis?: number): B;
  split(
    x: B,
    split_size_or_sections: number | number[],
    dim?: number
  ): B[];
}

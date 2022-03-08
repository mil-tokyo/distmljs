import { Backend } from '../../backend';
import {
  DType,
  DTypeDefault,
  TypedArrayForDType,
  TypedArrayTypes,
} from '../../dtype';
import { coreadd, corediv, coremul, corepow, coresub } from './core/binary';
import { broadcastTo, stridedCopy } from './core/copy';
import { argmax, argmin, max, min, sum, sumTo } from './core/reduction';
import {
  coreabs,
  coreacos,
  coreacosh,
  coreasin,
  coreasinh,
  coreatan,
  coreatanh,
  corecos,
  corecosh,
  coreexp,
  corelog,
  coreneg,
  corerelu,
  coresigmoid,
  coresin,
  coresinh,
  coresqrt,
  coresquare,
  coretan,
  coretanh,
  unaryWrap,
} from './core/unary';
import {
  calcReshape,
  calcSqueeze,
  calcTransposeShape,
  calcUnsqueeze,
  getMultiBroadcastShape,
} from '../shapeUtil';
import { Tensor } from '../tensor';
import { WebGLTensor } from '../webgl/webglTensor';
import { Ellipsis, Slice } from '..';
import { gets, sets } from './core/indexing';
import { WebGPUTensor } from '../webgpu/webgpuTensor';
import { cat, chunk, repeat, tile, split } from './core/manipulation';
import { gemm } from './core/gemm';
import { sort, topk } from './core/sort';
import { arrayProd } from '../../util';
import { tril, triu } from './core/tri';

class CPUTensorBuffer {
  public readonly data: TypedArrayTypes;
  constructor(public readonly length: number, public readonly dtype: DType) {
    this.data = new TypedArrayForDType[dtype](this.length);
  }
}

export type IndexingArg = number | Slice | Ellipsis | null;

export class CPUTensor extends Tensor {
  buffer: CPUTensorBuffer;

  private constructor(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: CPUTensorBuffer
  ) {
    super('cpu', shape, dtype);
    if (buffer) {
      this.buffer = buffer;
    } else {
      this.buffer = new CPUTensorBuffer(this.size, dtype);
    }
  }

  static isCPUTensor(tensor: unknown): tensor is CPUTensor {
    return typeof tensor === 'object' && (tensor as Tensor).backend === 'cpu';
  }

  getClass(): typeof CPUTensor {
    return CPUTensor;
  }

  alias(shape?: ArrayLike<number>): CPUTensor {
    return new CPUTensor(shape || this.shape, this.dtype, this.getBuffer());
  }

  to(backend: 'cpu'): Promise<CPUTensor>;
  to(backend: Backend): Promise<Tensor>;
  async to(backend: Backend): Promise<Tensor> {
    switch (backend) {
      case 'cpu':
        return this.alias();
      case 'webgl':
        return WebGLTensor.fromArray(this.toArray(), this.shape, this.dtype);
      case 'webgpu':
        return WebGPUTensor.fromArray(this.toArray(), this.shape, this.dtype);
      default:
        throw new Error(`Unknown backend ${backend}`);
    }
  }

  static zeros(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): CPUTensor {
    return new CPUTensor(shape, dtype);
  }

  static ones(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): CPUTensor {
    const t = new CPUTensor(shape, dtype);
    t.getBuffer().data.fill(1);
    return t;
  }

  static fromArray(
    data: ArrayLike<number>,
    shape?: ArrayLike<number>,
    dtype?: DType
  ): CPUTensor {
    const t = new CPUTensor(shape || [data.length], dtype);
    t.setArray(data);
    return t;
  }

  setArray(data: ArrayLike<number>): void {
    if (data.length !== this.size) {
      throw new Error('length mismatch');
    }
    this.buffer.data.set(data);
  }

  /**
   * Create CPUTensor from scalar
   * @param value scalar value
   * @returns CPUTensor of shape=[] (ndim=0)
   */
  static s(value: number): CPUTensor {
    const t = new CPUTensor([]);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    t.buffer!.data[0] = value;
    return t;
  }

  toTypedArray(): TypedArrayTypes {
    const buffer = this.getBuffer();
    return buffer.data.slice();
  }

  toArray(): number[] {
    const buffer = this.getBuffer();
    return Array.from(buffer.data);
  }

  async toArrayAsync(): Promise<number[]> {
    // GPUテンソルと同じインターフェースを提供する目的
    return this.toArray();
  }

  async toTypedArrayAsync(): Promise<TypedArrayTypes> {
    // GPUテンソルと同じインターフェースを提供する目的
    return this.toTypedArray();
  }

  getBuffer(): CPUTensorBuffer {
    if (!this.buffer) {
      throw new Error('CPUTensor already disposed');
    }
    return this.buffer;
  }

  get(...idxs: number[]): number {
    // TODO: negative index
    const buffer = this.getBuffer();
    let idx = 0;
    for (let i = 0; i < this.ndim; i++) {
      idx += this.strides[i] * (idxs[i] | 0);
    }

    return buffer.data[idx];
  }

  set(value: number, ...idxs: number[]): void {
    // TODO: negative index
    const buffer = this.getBuffer();
    let idx = 0;
    const nd = Math.min(this.ndim, idxs.length);
    for (let i = 0; i < nd; i++) {
      idx += this.strides[i] * (idxs[i] | 0);
    }

    buffer.data[idx] = value;
  }

  /**
   * Get slice of tensor. Currently supports basic indexing (as in numpy manual) only.
   * @param idxs index
   * @returns
   */
  gets(...idxs: IndexingArg[]): CPUTensor {
    return gets(this, idxs);
  }

  /**
   * Set value into a slice of tensor
   * @param value
   * @param idxs
   */
  sets(value: CPUTensor | number, ...idxs: IndexingArg[]): void {
    sets(this, value, idxs);
  }

  dispose() {
    if (!this.buffer) {
      throw new Error('Double-dispose of CPUTensor');
    }
    (this as { buffer: CPUTensorBuffer | null }).buffer = null;
  }

  copy(): CPUTensor {
    const dst = new CPUTensor(this.shape, this.dtype);
    dst.buffer.data.set(this.buffer.data);
    return dst;
  }

  static abs(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreabs);
  }

  static acos(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreacos);
  }

  static acosh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreacosh);
  }

  static asin(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreasin);
  }

  static asinh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreasinh);
  }

  static atan(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreatan);
  }

  static atanh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreatanh);
  }

  static cos(x: CPUTensor): CPUTensor {
    return unaryWrap(x, corecos);
  }

  static cosh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, corecosh);
  }

  static exp(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreexp);
  }

  static log(x: CPUTensor): CPUTensor {
    return unaryWrap(x, corelog);
  }

  static neg(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coreneg);
  }

  static relu(x: CPUTensor): CPUTensor {
    return unaryWrap(x, corerelu);
  }

  static sigmoid(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coresigmoid);
  }

  static sin(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coresin);
  }

  static sinh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coresinh);
  }

  static sqrt(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coresqrt);
  }

  static square(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coresquare);
  }

  static tan(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coretan);
  }

  static tanh(x: CPUTensor): CPUTensor {
    return unaryWrap(x, coretanh);
  }

  static add(lhs: CPUTensor, rhs: CPUTensor): CPUTensor {
    return coreadd(lhs, rhs);
  }

  static sub(lhs: CPUTensor, rhs: CPUTensor): CPUTensor {
    return coresub(lhs, rhs);
  }

  static mul(lhs: CPUTensor, rhs: CPUTensor): CPUTensor {
    return coremul(lhs, rhs);
  }

  static div(lhs: CPUTensor, rhs: CPUTensor): CPUTensor {
    return corediv(lhs, rhs);
  }

  static pow(lhs: CPUTensor, rhs: CPUTensor): CPUTensor {
    return corepow(lhs, rhs);
  }

  /**
   * 転置込みの行列積を行う暫定的な関数
   * @param a
   * @param b
   * @param transa
   * @param transb
   */
  static gemm(
    a: CPUTensor,
    b: CPUTensor,
    transa = false,
    transb = false
  ): CPUTensor {
    return gemm(a, b, transa, transb);
  }

  static dot(a: CPUTensor, b: CPUTensor): CPUTensor {
    return CPUTensor.gemm(a, b, false, false);
  }

  static broadcastTo(x: CPUTensor, shape: ReadonlyArray<number>): CPUTensor {
    return broadcastTo(x, shape);
  }

  static broadcastShapes(
    shapes: ReadonlyArray<ReadonlyArray<number>>
  ): number[] {
    const { shape } = getMultiBroadcastShape(shapes);
    return shape;
  }

  static sum(
    x: CPUTensor,
    axis?: number | number[] | null,
    keepdims?: boolean
  ): CPUTensor {
    return sum(x, axis, keepdims);
  }

  static sumTo(x: CPUTensor, shape: ReadonlyArray<number>): CPUTensor {
    return sumTo(x, shape);
  }

  static reshape(
    x: CPUTensor,
    shape: ReadonlyArray<number> | number,
    allowZero = true
  ): CPUTensor {
    return x.alias(calcReshape(x.shape, shape, allowZero));
  }

  reshape(shape: ReadonlyArray<number> | number, allowZero = true): CPUTensor {
    return CPUTensor.reshape(this, shape, allowZero);
  }

  static transpose(
    x: CPUTensor,
    axes?: ReadonlyArray<number> | null
  ): CPUTensor {
    const { newShape, srcStrides } = calcTransposeShape(
      x.shape,
      x.strides,
      axes
    );
    return stridedCopy(x, newShape, srcStrides);
  }

  transpose(axes?: ReadonlyArray<number> | null): CPUTensor {
    return CPUTensor.transpose(this, axes);
  }

  /**
   * Flatten to 1D array. Always returns copy.
   * @param x
   * @returns
   */
  static flatten(x: CPUTensor): CPUTensor {
    return CPUTensor.reshape(x, [-1]).copy();
  }

  /**
   * Flatten to 1D array. Always returns an alias to original tensor.
   * @param x
   * @returns
   */
  static ravel(x: CPUTensor): CPUTensor {
    return CPUTensor.reshape(x, [-1]);
  }

  static repeat(
    x: CPUTensor,
    repeats: ReadonlyArray<number> | number,
    axis?: number
  ): CPUTensor {
    return repeat(x, repeats, axis);
  }

  static tile(x: CPUTensor, reps: ReadonlyArray<number> | number): CPUTensor {
    return tile(x, reps);
  }

  static chunk(x: CPUTensor, chunks: number, dim?: number): CPUTensor[] {
    return chunk(x, chunks, dim);
  }

  static cat(tensors: ReadonlyArray<CPUTensor>, axis = 0): CPUTensor {
    return cat(tensors, axis);
  }

  static squeeze(input: CPUTensor, dim?: number): CPUTensor {
    return input.alias(calcSqueeze(input.shape, dim));
  }

  static unsqueeze(input: CPUTensor, dim: number): CPUTensor {
    return input.alias(calcUnsqueeze(input.shape, dim));
  }

  static sort(
    input: CPUTensor,
    dim = -1,
    descending = false
  ): [CPUTensor, CPUTensor] {
    return sort(input, dim, descending);
  }

  static full(
    shape: ArrayLike<number>,
    fillValue: number,
    dtype: DType = DTypeDefault
  ): CPUTensor {
    const data = new TypedArrayForDType[dtype](arrayProd(shape));
    data.fill(fillValue);
    return CPUTensor.fromArray(data, shape, dtype);
  }

  static split(
    x: CPUTensor,
    split_size_or_sections: number | number[],
    dim = 0
  ): CPUTensor[] {
    return split(x, split_size_or_sections, dim);
  }

  static max(input: CPUTensor): CPUTensor;
  static max(
    input: CPUTensor,
    dim: number,
    keepdim?: boolean
  ): [CPUTensor, CPUTensor];

  static max(
    input: CPUTensor,
    dim?: number,
    keepdim = false
  ): CPUTensor | [CPUTensor, CPUTensor] {
    return max(input, dim, keepdim);
  }

  static min(input: CPUTensor): CPUTensor;
  static min(
    input: CPUTensor,
    dim: number,
    keepdim?: boolean
  ): [CPUTensor, CPUTensor];

  static min(
    input: CPUTensor,
    dim?: number,
    keepdim = false
  ): CPUTensor | [CPUTensor, CPUTensor] {
    return min(input, dim, keepdim);
  }

  static argmax(input: CPUTensor): CPUTensor;
  static argmax(input: CPUTensor, dim: number, keepdim?: boolean): CPUTensor;

  static argmax(input: CPUTensor, dim?: number, keepdim = false): CPUTensor {
    return argmax(input, dim, keepdim);
  }

  static argmin(input: CPUTensor): CPUTensor;
  static argmin(input: CPUTensor, dim: number, keepdim?: boolean): CPUTensor;

  static argmin(input: CPUTensor, dim?: number, keepdim = false): CPUTensor {
    return argmin(input, dim, keepdim);
  }

  static tril(input: CPUTensor, diagonal = 0): CPUTensor {
    return tril(input, diagonal);
  }

  static triu(input: CPUTensor, diagonal = 0): CPUTensor {
    return triu(input, diagonal);
  }

  static topk(
    input: CPUTensor,
    k: number,
    dim = -1,
    largest = true
  ): [CPUTensor, CPUTensor] {
    return topk(input, k, dim, largest);
  }
}

import { Backend } from '../../backend';
import {
  DType,
  DTypeDefault,
  TypedArrayForDType,
  TypedArrayTypes,
} from '../../dtype';
import { arrayEqual } from '../../util';
import { coreadd, corediv, coremul, corepow, coresub } from './core/binary';
import { broadcastTo, stridedCopy } from './core/copy';
import { sum, sumTo } from './core/reduction';
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
  calcTransposeShape,
  getMultiBroadcastShape,
} from '../shapeUtil';
import { Tensor } from '../tensor';
import { WebGLTensor } from '../webgl/webglTensor';
import { Ellipsis, Slice } from '..';
import { gets, sets } from './core/indexing';
import { WebGPUTensor } from '../webgpu/webgpuTensor';

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

  static reluBackprop(x: CPUTensor, gy: CPUTensor): CPUTensor {
    const output = CPUTensor.zeros(x.shape);
    const dx = x.getBuffer().data;
    const dgy = gy.getBuffer().data;
    const dgx = output.getBuffer().data;
    for (let i = 0; i < output.size; i++) {
      dgx[i] = dx[i] > 0.0 ? dgy[i] : 0.0;
    }
    return output;
  }

  static sigmoidBackprop(y: CPUTensor, gy: CPUTensor): CPUTensor {
    const output = CPUTensor.zeros(gy.shape);
    const dy = y.getBuffer().data;
    const dgy = gy.getBuffer().data;
    const dgx = output.getBuffer().data;
    for (let i = 0; i < output.size; i++) {
      const yv = dy[i];
      dgx[i] = (1 - yv) * yv * dgy[i];
    }
    return output;
  }

  static mseLoss(a: CPUTensor, b: CPUTensor): CPUTensor {
    if (!arrayEqual(a.shape, b.shape)) {
      throw new Error('Shape mismatch');
    }
    const output = CPUTensor.zeros([]);
    const da = a.getBuffer().data;
    const db = b.getBuffer().data;
    const dy = output.getBuffer().data;
    let sum = 0.0;
    for (let i = 0; i < a.size; i++) {
      const diff = da[i] - db[i];
      sum += diff * diff;
    }
    dy[0] = sum / a.size;
    return output;
  }

  static mseLossBackprop(
    a: CPUTensor,
    b: CPUTensor,
    gy: CPUTensor
  ): [CPUTensor, CPUTensor] {
    if (!arrayEqual(a.shape, b.shape)) {
      throw new Error('Shape mismatch');
    }
    if (gy.ndim !== 0) {
      throw new Error('gy must be scalar');
    }
    const da = a.getBuffer().data;
    const db = b.getBuffer().data;
    const dgy = gy.getBuffer().data;
    const ga = CPUTensor.zeros(a.shape);
    const gb = CPUTensor.zeros(a.shape);
    const dga = ga.getBuffer().data;
    const dgb = gb.getBuffer().data;
    const coef = (dgy[0] * 2) / a.size;
    for (let i = 0; i < a.size; i++) {
      const v = (da[i] - db[i]) * coef;
      dga[i] = v;
      dgb[i] = -v;
    }
    return [ga, gb];
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
    let m: number, n: number, k: number, bk: number;
    let stam: number, stak: number, stbk: number, stbn: number; //strides
    if (a.ndim !== 2 || b.ndim !== 2) {
      throw new Error('must be 2dim');
    }
    if (transa) {
      [k, m] = a.shape;
      [stak, stam] = a.strides;
    } else {
      [m, k] = a.shape;
      [stam, stak] = a.strides;
    }
    if (transb) {
      [n, bk] = b.shape;
      [stbn, stbk] = b.strides;
    } else {
      [bk, n] = b.shape;
      [stbk, stbn] = b.strides;
    }
    if (k !== bk) {
      throw new Error('inner product length does not match');
    }

    const output = CPUTensor.zeros([m, n]);
    let i = 0;
    const da = a.getBuffer().data;
    const db = b.getBuffer().data;
    const dy = output.getBuffer().data;
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let sum = 0.0;
        for (let ip = 0; ip < k; ip++) {
          sum += da[row * stam + ip * stak] * db[col * stbn + ip * stbk];
        }
        dy[i++] = sum;
      }
    }
    return output;
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

  static softmax(x: CPUTensor): CPUTensor {
    const [batch, cs] = x.shape;
    if (x.shape.length !== 2) {
      throw new Error('softmaxCrossEntropy needs 2d input');
    }
    const output = CPUTensor.zeros([batch, cs]);
    const dx = x.getBuffer().data;
    const dy = output.getBuffer().data;
    for (let b = 0; b < batch; b++) {
      let max = -Infinity;
      for (let c = 0; c < cs; c++) {
        const v = dx[b * cs + c];
        if (v > max) {
          max = v;
        }
      }
      let expSum = 0.0;
      for (let c = 0; c < cs; c++) {
        const v = dx[b * cs + c] - max;
        const exp = Math.exp(v);
        dy[b * cs + c] = exp;
        expSum += exp;
      }
      for (let c = 0; c < cs; c++) {
        dy[b * cs + c] /= expSum;
      }
    }
    return output;
  }

  static nllLoss(x: CPUTensor, label: CPUTensor): CPUTensor {
    const [batch, cs] = x.shape;
    if (x.shape.length !== 2) {
      throw new Error('nllLoss needs 2d input');
    }
    if (label.shape.length !== 1) {
      throw new Error('nllLoss needs 1d label input');
    }
    const output = CPUTensor.zeros([]);
    const dx = x.getBuffer().data;
    const dlabel = label.getBuffer().data;
    let ceSum = 0.0;
    for (let b = 0; b < batch; b++) {
      const label = dlabel[b];
      ceSum += -Math.log(dx[b * cs + label]);
    }
    const ceAvg = ceSum / batch;
    output.getBuffer().data[0] = ceAvg;
    return output;
  }

  static softmaxCrossEntropyBackward(
    softmax: CPUTensor,
    label: CPUTensor,
    gy: CPUTensor
  ): CPUTensor {
    // x -> softmax -> lossでgxを求める
    // TODO: labelはint32
    const [batch, cs] = softmax.shape;
    if (softmax.shape.length !== 2) {
      throw new Error('nllLoss needs 2d input');
    }
    if (label.shape.length !== 1) {
      throw new Error('nllLoss needs 1d label input');
    }
    if (!arrayEqual(gy.shape, [])) {
      throw new Error('gy must be scalar');
    }
    const output = CPUTensor.zeros(softmax.shape);
    const dgx = output.getBuffer().data;
    const dsoftmax = softmax.getBuffer().data;
    const dlabel = label.getBuffer().data;
    const dgy = gy.getBuffer().data;
    const gyValue = dgy[0] / batch;
    for (let b = 0; b < batch; b++) {
      const label = dlabel[b];
      for (let c = 0; c < cs; c++) {
        let v = dsoftmax[b * cs + c];
        if (c === label) {
          v -= 1;
        }
        dgx[b * cs + c] = v * gyValue;
      }
    }
    return output;
  }

  static reshape(
    x: CPUTensor,
    shape: ReadonlyArray<number> | number,
    allowZero = true
  ): CPUTensor {
    return x.alias(calcReshape(x.shape, shape, allowZero));
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
}

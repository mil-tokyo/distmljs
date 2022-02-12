import { Backend } from '../../backend';
import {
  DType,
  DTypeDefault,
  TypedArrayConstructor,
  TypedArrayTypes,
} from '../../dtype';
import { arrayProd } from '../../util';
import { CPUTensor } from '../cpu/cpuTensor';
import {
  calcReshape,
  calcTransposeShape,
  getBroadcastStride,
} from '../shapeUtil';
import { Tensor } from '../tensor';
import {
  coreadd,
  corediv,
  coremul,
  corepow,
  corereluBackprop,
  coresigmoidBackprop,
  coresub,
} from './core/binary';
import { stridedCopy } from './core/copy';
import {
  mseLoss,
  mseLossBackprop,
  nllLoss,
  softmax,
  softmaxCrossEntropyBackward,
} from './core/loss';
import { sum, sumTo } from './core/reduction';
import { gemm } from './core/standard';
import {
  coreabs,
  coreacos,
  coreacosh,
  coreasin,
  coreasinh,
  coreatan,
  coreatanh,
  corecopy,
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
} from './core/unary';
import { getNNWebGPUContext } from './webgpuContext';

let webgpuAllocCount = 0;
export const existingBuffers: Set<WebGPUTensorBuffer> = new Set();

type TypedArrayTypesForWebGPUBuffer = Float32Array | Int32Array | Uint32Array;

export interface WebGPUBufferShape {
  byteLength: number;
  forWriteFromCPU: boolean;
  forReadToCPU: boolean;
}

export class WebGPUTensorBuffer {
  public ref: number;
  gpuBuffer: GPUBuffer;

  private mappedForWriteFromCPU: boolean;

  constructor(
    public readonly bufferShape: WebGPUBufferShape,
    public readonly forMetaBuffer = false
  ) {
    const ctx = getNNWebGPUContext();
    this.ref = 1;
    let usage = GPUBufferUsage.STORAGE;
    if (bufferShape.forReadToCPU) {
      usage |= GPUBufferUsage.COPY_SRC;
    }
    this.gpuBuffer = ctx.device.createBuffer({
      mappedAtCreation: bufferShape.forWriteFromCPU,
      size: bufferShape.byteLength,
      usage,
    });
    this.mappedForWriteFromCPU = bufferShape.forWriteFromCPU;
    // meta bufferは別に管理する
    if (!this.forMetaBuffer) {
      webgpuAllocCount++;
      existingBuffers.add(this);
    }
  }

  setDataRaw(data: TypedArrayTypesForWebGPUBuffer): void {
    if (!this.mappedForWriteFromCPU) {
      // TODO: enable write again by creating temporary buffer and copybuffertobuffer
      throw new Error(
        'The buffer is not mapped. WebGPUTensor can only be written just after creation.'
      );
    }

    const ab = this.gpuBuffer.getMappedRange();
    // create same typedarray as data
    const mappedArray = new (data.constructor as TypedArrayConstructor)(ab);
    mappedArray.set(data);
    this.gpuBuffer.unmap();
    this.mappedForWriteFromCPU = false;
  }

  async getDataRaw(dtype: DType): Promise<TypedArrayTypesForWebGPUBuffer> {
    if (!this.bufferShape.forReadToCPU) {
      throw new Error(
        'forReadToCPU flag is not set for this WebGPUTensor. Please use WebGPUTensor.copy() to create readable tensor.'
      );
    }
    const ctx = getNNWebGPUContext();
    let ctor: typeof Float32Array | typeof Int32Array | typeof Uint32Array;
    let itemCount: number;
    switch (dtype) {
      case 'float32':
        ctor = Float32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'int32':
        ctor = Int32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'uint8':
      case 'bool':
        // Uint8ArrayではなくUint32Arrayで格納されている
        ctor = Uint32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      default:
        throw new Error(`Unknown dtype ${dtype}`);
    }

    const data: TypedArrayTypesForWebGPUBuffer = new ctor(itemCount),
      dst = ctx.device.createBuffer({
        size: this.bufferShape.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      commandEncoder = ctx.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.gpuBuffer,
      0,
      dst,
      0,
      this.bufferShape.byteLength
    );
    ctx.device.queue.submit([commandEncoder.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const arrayBuffer = dst.getMappedRange(),
      buffer_mapped_array = new ctor(arrayBuffer, 0, itemCount);
    data.set(buffer_mapped_array);
    dst.unmap();
    dst.destroy();
    return data;
  }

  dispose() {
    this.gpuBuffer.destroy();
    if (!this.forMetaBuffer) {
      webgpuAllocCount--;
      existingBuffers.delete(this);
    }
    (this as { gpuBuffer: GPUBuffer | null }).gpuBuffer = null;
  }
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function calcDefaultBufferShape(size: number, dtype: DType): WebGPUBufferShape {
  // glslのlayoutがbyte単位の要素をサポートしておらず、uint8/boolでも1要素32bitのuintとして扱う
  const bytePerElement = 4;
  let byteLength = size * bytePerElement;
  // 将来4の倍数でない要素をサポートする場合、4の倍数に切り上げる必要あり
  // 0バイトはエラーとなるため4バイトとする。
  if (byteLength < 4) {
    byteLength = 4;
  }
  return { byteLength, forWriteFromCPU: false, forReadToCPU: false };
}

export class WebGPUTensor extends Tensor {
  buffer: WebGPUTensorBuffer;

  constructor(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: WebGPUTensorBuffer,
    bufferShape?: WebGPUBufferShape
  ) {
    super('webgpu', shape, dtype);
    const bShape = bufferShape || calcDefaultBufferShape(this.size, this.dtype);
    if (bShape.forWriteFromCPU && bShape.forReadToCPU) {
      throw new Error('WebGPUTensor cannot be both for read and write');
    }
    this.buffer = buffer || new WebGPUTensorBuffer(bShape);
  }

  static isWebGPUTensor(tensor: unknown): tensor is WebGPUTensor {
    return (
      typeof tensor === 'object' && (tensor as Tensor).backend === 'webgpu'
    );
  }

  getClass(): typeof WebGPUTensor {
    return WebGPUTensor;
  }

  alias(shape?: ArrayLike<number>): WebGPUTensor {
    let t: WebGPUTensor;
    const buffer = this.buffer;
    try {
      buffer.ref++;
      t = new WebGPUTensor(shape || this.shape, this.dtype, buffer);
    } catch (error) {
      buffer.ref--;
      throw error;
    }
    return t;
  }

  static getDebugInfo() {
    return { webgpuAllocCount };
  }

  static s(value: number): WebGPUTensor {
    return WebGPUTensor.fromArray([value], []);
  }

  static empty(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: WebGPUTensorBuffer,
    bufferShape?: WebGPUBufferShape
  ): WebGPUTensor {
    let bShape: WebGPUBufferShape;
    if (bufferShape) {
      bShape = bufferShape;
    } else {
      bShape = calcDefaultBufferShape(arrayProd(shape), dtype);
      // CPUへの読み取りに使われると仮定
      bShape.forReadToCPU = true;
    }
    return new WebGPUTensor(shape, dtype, buffer, bShape);
  }

  static zeros(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): WebGPUTensor {
    const data = new Float32Array(arrayProd(shape));
    return WebGPUTensor.fromArray(data, shape, dtype);
  }

  static ones(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): WebGPUTensor {
    const data = new Float32Array(arrayProd(shape));
    data.fill(1);
    return WebGPUTensor.fromArray(data, shape, dtype);
  }

  to(backend: 'cpu'): Promise<CPUTensor>;
  to(backend: Backend): Promise<Tensor>;
  async to(backend: Backend): Promise<Tensor> {
    switch (backend) {
      case 'cpu':
        return CPUTensor.fromArray(
          await this.toTypedArrayAsync(),
          this.shape,
          this.dtype
        );
      case 'webgpu':
        return this.alias();
      default:
        throw new Error(`Unknown backend ${backend}`);
    }
  }

  static fromArray(
    data: ArrayLike<number>,
    shape?: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): WebGPUTensor {
    const shape_ = shape || [data.length];
    const bShape = calcDefaultBufferShape(arrayProd(shape_), dtype);
    bShape.forWriteFromCPU = true;
    const t = new WebGPUTensor(shape_, dtype, undefined, bShape);
    t.setArray(data);
    return t;
  }

  setArray(data: ArrayLike<number>): void {
    if (data.length !== this.size) {
      throw new Error('length mismatch');
    }
    let ctor: typeof Float32Array | typeof Int32Array | typeof Uint32Array;
    let itemCount: number;
    switch (this.dtype) {
      case 'float32':
        ctor = Float32Array;
        itemCount = this.buffer.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'int32':
        ctor = Int32Array;
        itemCount = this.buffer.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'uint8':
      case 'bool':
        ctor = Uint32Array;
        itemCount = this.buffer.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      default:
        throw new Error(`Unknown dtype ${this.dtype}`);
    }
    const packed = new ctor(itemCount);
    packed.set(data);
    this.buffer.setDataRaw(packed);
  }

  async toTypedArrayAsync(): Promise<TypedArrayTypes> {
    const rawData = await this.buffer.getDataRaw(this.dtype);

    // rawDataの型はあっていても、長さが余分な場合があるのでViewを作る
    let view: TypedArrayTypes;
    switch (this.dtype) {
      case 'float32':
        view = new Float32Array(rawData.buffer, rawData.byteOffset, this.size);
        break;
      case 'int32':
        view = new Int32Array(rawData.buffer, rawData.byteOffset, this.size);
        break;
      case 'uint8':
      case 'bool':
        {
          view = new Uint8Array(this.size);
          const uint32View = new Uint32Array(
            rawData.buffer,
            rawData.byteOffset,
            this.size
          );
          view.set(uint32View);
        }
        break;
      default:
        throw new Error(`Unknown dtype ${this.dtype}`);
    }
    return view;
  }

  async toArrayAsync(): Promise<number[]> {
    return Array.from(await this.toTypedArrayAsync());
  }

  dispose() {
    if (!this.buffer) {
      throw new Error('Double-dispose of WebGPUTensor');
    }
    this.buffer.ref--;
    if (this.buffer.ref <= 0) {
      this.buffer.dispose();
    }
    (this as { buffer: WebGPUTensorBuffer | null }).buffer = null;
  }

  copy(): WebGPUTensor {
    return corecopy(this);
  }

  static abs(x: WebGPUTensor): WebGPUTensor {
    return coreabs(x);
  }

  static acos(x: WebGPUTensor): WebGPUTensor {
    return coreacos(x);
  }

  static acosh(x: WebGPUTensor): WebGPUTensor {
    return coreacosh(x);
  }

  static asin(x: WebGPUTensor): WebGPUTensor {
    return coreasin(x);
  }

  static asinh(x: WebGPUTensor): WebGPUTensor {
    return coreasinh(x);
  }

  static atan(x: WebGPUTensor): WebGPUTensor {
    return coreatan(x);
  }

  static atanh(x: WebGPUTensor): WebGPUTensor {
    return coreatanh(x);
  }

  static cos(x: WebGPUTensor): WebGPUTensor {
    return corecos(x);
  }

  static cosh(x: WebGPUTensor): WebGPUTensor {
    return corecosh(x);
  }

  static exp(x: WebGPUTensor): WebGPUTensor {
    return coreexp(x);
  }

  static log(x: WebGPUTensor): WebGPUTensor {
    return corelog(x);
  }

  static neg(x: WebGPUTensor): WebGPUTensor {
    return coreneg(x);
  }

  static relu(x: WebGPUTensor): WebGPUTensor {
    return corerelu(x);
  }

  static sigmoid(x: WebGPUTensor): WebGPUTensor {
    return coresigmoid(x);
  }

  static sin(x: WebGPUTensor): WebGPUTensor {
    return coresin(x);
  }

  static sinh(x: WebGPUTensor): WebGPUTensor {
    return coresinh(x);
  }

  static sqrt(x: WebGPUTensor): WebGPUTensor {
    return coresqrt(x);
  }

  static square(x: WebGPUTensor): WebGPUTensor {
    return coresquare(x);
  }

  static tan(x: WebGPUTensor): WebGPUTensor {
    return coretan(x);
  }

  static tanh(x: WebGPUTensor): WebGPUTensor {
    return coretanh(x);
  }

  static add(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return coreadd(lhs, rhs);
  }

  static sub(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return coresub(lhs, rhs);
  }

  static mul(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return coremul(lhs, rhs);
  }

  static div(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return corediv(lhs, rhs);
  }

  static pow(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return corepow(lhs, rhs);
  }

  static sigmoidBackprop(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return coresigmoidBackprop(lhs, rhs);
  }

  static reluBackprop(lhs: WebGPUTensor, rhs: WebGPUTensor): WebGPUTensor {
    return corereluBackprop(lhs, rhs);
  }

  /**
   * 転置込みの行列積を行う暫定的な関数
   * @param a
   * @param b
   * @param transa
   * @param transb
   */
  static gemm(
    a: WebGPUTensor,
    b: WebGPUTensor,
    transa = false,
    transb = false
  ): WebGPUTensor {
    return gemm(a, b, transa, transb);
  }

  static dot(a: WebGPUTensor, b: WebGPUTensor): WebGPUTensor {
    return WebGPUTensor.gemm(a, b, false, false);
  }

  static broadcastTo(
    x: WebGPUTensor,
    shape: ReadonlyArray<number>
  ): WebGPUTensor {
    const xStride = getBroadcastStride(x.shape, shape);
    return stridedCopy(x, shape, xStride);
  }

  static sum(
    x: WebGPUTensor,
    axis?: number | number[] | null,
    keepdims?: boolean
  ): WebGPUTensor {
    return sum(x, axis, keepdims);
  }

  static sumTo(x: WebGPUTensor, shape: ReadonlyArray<number>): WebGPUTensor {
    return sumTo(x, shape);
  }

  static reshape(
    x: WebGPUTensor,
    shape: ReadonlyArray<number> | number,
    allowZero = true
  ): WebGPUTensor {
    return x.alias(calcReshape(x.shape, shape, allowZero));
  }

  reshape(
    shape: ReadonlyArray<number> | number,
    allowZero = true
  ): WebGPUTensor {
    return WebGPUTensor.reshape(this, shape, allowZero);
  }

  static transpose(
    x: WebGPUTensor,
    axes?: ReadonlyArray<number> | null
  ): WebGPUTensor {
    const { newShape, srcStrides } = calcTransposeShape(
      x.shape,
      x.strides,
      axes
    );
    return stridedCopy(x, newShape, srcStrides);
  }

  transpose(axes?: ReadonlyArray<number> | null): WebGPUTensor {
    return WebGPUTensor.transpose(this, axes);
  }

  static mseLossBackprop(
    ad: WebGPUTensor,
    bd: WebGPUTensor,
    gyd: WebGPUTensor
  ): [WebGPUTensor, WebGPUTensor] {
    return mseLossBackprop(ad, bd, gyd);
  }

  static mseLoss(a: WebGPUTensor, b: WebGPUTensor): WebGPUTensor {
    return mseLoss(a, b);
  }

  static nllLoss(softmax: WebGPUTensor, label: WebGPUTensor): WebGPUTensor {
    return nllLoss(softmax, label);
  }

  static softmax(x: WebGPUTensor): WebGPUTensor {
    return softmax(x);
  }

  static softmaxCrossEntropyBackward(
    softmax: WebGPUTensor,
    label: WebGPUTensor,
    gy: WebGPUTensor
  ): WebGPUTensor {
    return softmaxCrossEntropyBackward(softmax, label, gy);
  }

  /**
   * Flatten to 1D array. Always returns copy.
   * @param x
   * @returns
   */
  static flatten(x: WebGPUTensor): WebGPUTensor {
    return WebGPUTensor.reshape(x, [-1]).copy();
  }

  /**
   * Flatten to 1D array. Always returns an alias to original tensor.
   * @param x
   * @returns
   */
  static ravel(x: WebGPUTensor): WebGPUTensor {
    return WebGPUTensor.reshape(x, [-1]);
  }
}

import { Backend } from '../../backend';
import {
  DType,
  DTypeDefault,
  TypedArrayForDType,
  TypedArrayTypes,
} from '../../dtype';
import { arrayEqual } from '../../util';
import { CPUTensor } from '../cpuTensor';
import { Tensor } from '../tensor';

let webglAllocCount = 0;
const existingBuffers: Set<WebGLTensorBuffer> = new Set();

class WebGLTensorBuffer {
  // TODO: GPUアクセスの実装
  public readonly mem: TypedArrayTypes;
  public ref: number;
  constructor(public readonly length: number, public readonly dtype: DType) {
    this.ref = 1;
    this.mem = new TypedArrayForDType[dtype](this.length);
    webglAllocCount++;
    existingBuffers.add(this);
  }

  dispose() {
    // TODO: dispose gpu memory
    webglAllocCount--;
    existingBuffers.delete(this);
  }
}

export class WebGLTensor extends Tensor {
  buffer: WebGLTensorBuffer | null;

  private constructor(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: WebGLTensorBuffer
  ) {
    super('webgl', shape, dtype);
    if (buffer) {
      this.buffer = buffer;
    } else {
      this.buffer = new WebGLTensorBuffer(this.size, dtype);
    }
  }

  alias(shape?: ArrayLike<number>): WebGLTensor {
    let t: WebGLTensor;
    const buffer = this.getBuffer();
    try {
      buffer.ref++;
      t = new WebGLTensor(shape || this.shape, this.dtype, buffer);
    } catch (error) {
      buffer.ref--;
      throw error;
    }
    return t;
  }

  static getDebugInfo() {
    return { webglAllocCount };
  }

  static async zeros(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): Promise<WebGLTensor> {
    return new WebGLTensor(shape, dtype);
  }

  static async ones(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): Promise<WebGLTensor> {
    const t = new WebGLTensor(shape, dtype);
    t.getBuffer().mem.fill(1);
    return t;
  }

  to(backend: 'cpu'): Promise<CPUTensor>;
  async to(backend: Backend): Promise<Tensor> {
    switch (backend) {
      case 'cpu':
        return CPUTensor.fromArray(await this.toArray(), this.shape);
      case 'webgl':
        return this.alias();
      default:
        throw new Error(`Unknown backend ${backend}`);
    }
  }

  static async fromArray(
    data: ArrayLike<number>,
    shape?: ArrayLike<number>
  ): Promise<WebGLTensor> {
    const t = new WebGLTensor(shape || [data.length]);
    if (data.length !== t.size) {
      throw new Error('length mismatch');
    }
    t.buffer?.mem.set(data);
    return t;
  }

  async toArray(): Promise<number[]> {
    const buffer = this.getBuffer();
    return Array.from(buffer.mem);
  }

  private getBuffer(): WebGLTensorBuffer {
    if (!this.buffer) {
      throw new Error('WebGLTensor already disposed');
    }
    return this.buffer;
  }

  dispose() {
    if (!this.buffer) {
      throw new Error('Double-dispose of WebGLTensor');
    }
    this.buffer.ref--;
    if (this.buffer.ref <= 0) {
      this.buffer.dispose();
    }
    this.buffer = null;
  }

  static add(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    // TODO: broadcast, type check
    if (!arrayEqual(lhs.shape, rhs.shape)) {
      throw new Error('shape not match');
    }
    const output = new WebGLTensor(lhs.shape);
    const dl = lhs.getBuffer().mem;
    const dr = rhs.getBuffer().mem;
    const dy = output.getBuffer().mem;
    for (let i = 0; i < output.size; i++) {
      dy[i] = dl[i] + dr[i];
    }
    return output;
  }

  static mul(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    // TODO: broadcast, type check
    if (!arrayEqual(lhs.shape, rhs.shape)) {
      throw new Error('shape not match');
    }
    const output = new WebGLTensor(lhs.shape);
    const dl = lhs.getBuffer().mem;
    const dr = rhs.getBuffer().mem;
    const dy = output.getBuffer().mem;
    for (let i = 0; i < output.size; i++) {
      dy[i] = dl[i] * dr[i];
    }
    return output;
  }

  static exp(x: WebGLTensor): WebGLTensor {
    const output = new WebGLTensor(x.shape);
    const dx = x.getBuffer().mem;
    const dy = output.getBuffer().mem;
    for (let i = 0; i < output.size; i++) {
      dy[i] = Math.exp(dx[i]);
    }
    return output;
  }
}

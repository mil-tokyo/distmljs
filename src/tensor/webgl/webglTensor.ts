import { Backend } from '../../backend';
import {
  DType,
  DTypeDefault,
  TypedArrayForDType,
  TypedArrayTypes,
} from '../../dtype';
import { arrayEqual, arrayProd } from '../../util';
import { CPUTensor } from '../cpu/cpuTensor';
import { Tensor } from '../tensor';
import { add } from './core/binary';
import {
  packToFloat16Array,
  packToFloat32Array,
  packToInt32Array,
  packToUint8Array,
  unpackFromFloat16Array,
  unpackFromFloat32Array,
  unpackFromInt32Array,
  unpackFromUint8Array,
} from './core/pack';
import { exp } from './core/unary';
import { getNNWebGLContext, webglShaderHeader } from './webglContext';

let webglAllocCount = 0;
const existingBuffers: Set<WebGLTensorBuffer> = new Set();
export interface TensorTextureShapeFormat {
  internalFormat: number;
  format: number;
  type: number;
}

export const tensorTextureShapeFormatR32F = {
  internalFormat: WebGL2RenderingContext.R32F,
  format: WebGL2RenderingContext.RED,
  type: WebGL2RenderingContext.FLOAT,
};

export const tensorTextureShapeFormatR16F = {
  internalFormat: WebGL2RenderingContext.R16F,
  format: WebGL2RenderingContext.RED,
  type: WebGL2RenderingContext.HALF_FLOAT,
};

export const tensorTextureShapeFormatR32I = {
  internalFormat: WebGL2RenderingContext.R32I,
  format: WebGL2RenderingContext.RED_INTEGER,
  type: WebGL2RenderingContext.INT,
};

export const tensorTextureShapeFormatR8UI = {
  internalFormat: WebGL2RenderingContext.R8UI,
  format: WebGL2RenderingContext.RED_INTEGER,
  type: WebGL2RenderingContext.UNSIGNED_BYTE,
};

export const tensorTextureShapeFormatRGBA32F = {
  internalFormat: WebGL2RenderingContext.RGBA32F,
  format: WebGL2RenderingContext.RGBA,
  type: WebGL2RenderingContext.FLOAT,
};

export const tensorTextureShapeFormatRGBA16F = {
  internalFormat: WebGL2RenderingContext.RGBA16F,
  format: WebGL2RenderingContext.RGBA,
  type: WebGL2RenderingContext.HALF_FLOAT,
};

export const tensorTextureShapeFormatRGBA32I = {
  internalFormat: WebGL2RenderingContext.RGBA32I,
  format: WebGL2RenderingContext.RGBA,
  type: WebGL2RenderingContext.INT,
};

export function getTensorTextureShapeFormatForDType(
  dtype: DType,
  supportsTexture32bit?: boolean
): TensorTextureShapeFormat {
  let b32: boolean;
  if (supportsTexture32bit == null) {
    const context = getNNWebGLContext();
    b32 = context.supportsTexture32bit;
  } else {
    b32 = supportsTexture32bit;
  }
  let format: TensorTextureShapeFormat;
  switch (dtype) {
    case 'float32':
      format = b32
        ? tensorTextureShapeFormatR32F
        : tensorTextureShapeFormatR16F;
      break;
    case 'int32':
      format = tensorTextureShapeFormatR32I;
      break;
    case 'uint8':
      format = tensorTextureShapeFormatR8UI;
      break;
    case 'bool':
      format = tensorTextureShapeFormatR8UI;
      break;
    default:
      throw new Error(`WebGL texture for dtype ${dtype} is not yet supported`);
  }
  return format;
}

export const tensorTextureShapeFormatDefault = tensorTextureShapeFormatR32F;

export interface TensorTextureShape2D extends TensorTextureShapeFormat {
  dim: '2D';
  width: number;
  height: number;
}

export interface TensorTextureShape2DArray extends TensorTextureShapeFormat {
  dim: '2DArray';
  width: number;
  height: number;
  depth: number;
}

export type TensorTextureShape =
  | TensorTextureShape2D
  | TensorTextureShape2DArray;

class WebGLTensorBuffer {
  public readonly texture: WebGLTexture;
  public ref: number;
  public target: number;
  private isBoundToDrawFrameBuffer = false;
  private readTextureUnitIndices: number[] = [];
  public dimPerPixel: number;
  public textureLength: number;
  constructor(public readonly textureShape: TensorTextureShape) {
    this.ref = 1;
    const ctx = getNNWebGLContext();
    this.texture = ctx.createTexture(textureShape);
    switch (textureShape.format) {
      case WebGL2RenderingContext.RED:
      case WebGL2RenderingContext.RED_INTEGER:
        this.dimPerPixel = 1;
        break;
      case WebGL2RenderingContext.RGBA:
        this.dimPerPixel = 4;
        break;
      default:
        throw new Error();
    }
    switch (textureShape.dim) {
      case '2D':
        this.target = WebGL2RenderingContext.TEXTURE_2D;
        this.textureLength =
          textureShape.height * textureShape.width * this.dimPerPixel;
        break;
      case '2DArray':
        this.target = WebGL2RenderingContext.TEXTURE_2D_ARRAY;
        this.textureLength =
          textureShape.depth *
          textureShape.height *
          textureShape.width *
          this.dimPerPixel;
        break;
    }

    webglAllocCount++;
    existingBuffers.add(this);
  }

  dispose() {
    // TODO: texture pool
    webglAllocCount--;
    existingBuffers.delete(this);
    const ctx = getNNWebGLContext();
    ctx.gl.deleteTexture(this.texture);
    (this as { texture: WebGLTexture | null }).texture = null;
  }

  bindToReadTexture(unit: number): void {
    if (this.isBoundToDrawFrameBuffer)
      throw Error(
        'This buffer is already registered as draw buffer. ' +
          'You may forgot to unbind the binding while previous operations.'
      );

    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(this.target, this.texture);

    this.readTextureUnitIndices.push(unit);
  }

  unbindFromReadTexture(): void {
    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    for (const unit of this.readTextureUnitIndices) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(this.target, null);
    }

    this.readTextureUnitIndices = [];
  }

  bindToDrawTexture(layer = 0): void {
    if (this.readTextureUnitIndices.length > 0)
      throw Error(
        'This buffer is already registered as read buffer. ' +
          'You cannot bind a texture as both read and draw texture buffer at same time.'
      );
    if (this.isBoundToDrawFrameBuffer)
      throw Error(
        'This buffer is already registered as draw buffer. ' +
          'You may forgot to unbind the binding while previous operations.'
      );

    const ctx = getNNWebGLContext();
    const { gl } = ctx;
    gl.viewport(0, 0, this.textureShape.width, this.textureShape.height);
    gl.scissor(0, 0, this.textureShape.width, this.textureShape.height);

    switch (this.textureShape.dim) {
      case '2D':
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          this.texture,
          0
        );
        break;
      case '2DArray':
        gl.framebufferTextureLayer(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          this.texture,
          0,
          layer
        );
        break;
      default:
        throw new Error();
    }

    this.isBoundToDrawFrameBuffer = true;
  }

  unbindFromDrawTexture(): void {
    if (!this.isBoundToDrawFrameBuffer) return;

    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      null,
      0
    );

    this.isBoundToDrawFrameBuffer = false;
  }

  getDataRawFloat32(): Float32Array {
    if (
      this.textureShape.dim !== '2D' ||
      this.textureShape.type !== WebGL2RenderingContext.FLOAT
    ) {
      throw new Error();
    }
    const buf = new Float32Array(
      this.textureShape.height * this.textureShape.width * this.dimPerPixel
    );
    this.readPixels2D(buf);
    return buf;
  }

  getDataRaw():
    | { type: 'Float32Array'; buffer: Float32Array }
    | { type: 'Uint16Array'; buffer: Uint16Array }
    | { type: 'Int32Array'; buffer: Int32Array }
    | { type: 'Uint8Array'; buffer: Uint8Array } {
    switch (this.textureShape.dim) {
      case '2D': {
        const length =
          this.textureShape.height * this.textureShape.width * this.dimPerPixel;
        switch (this.textureShape.type) {
          case WebGL2RenderingContext.FLOAT: {
            const buffer = new Float32Array(length);
            this.readPixels2D(buffer);
            return { type: 'Float32Array', buffer };
          }
          case WebGL2RenderingContext.HALF_FLOAT: {
            const buffer = new Uint16Array(length);
            this.readPixels2D(buffer);
            return { type: 'Uint16Array', buffer };
          }
          case WebGL2RenderingContext.INT: {
            const buffer = new Int32Array(length);
            this.readPixels2D(buffer);
            return { type: 'Int32Array', buffer };
          }
          case WebGL2RenderingContext.UNSIGNED_BYTE: {
            const buffer = new Uint8Array(length);
            this.readPixels2D(buffer);
            return { type: 'Uint8Array', buffer };
          }
          default:
            throw new Error();
        }
      }
      case '2DArray': {
        const sliceLength =
          this.textureShape.height * this.textureShape.width * this.dimPerPixel;
        const totalLength = sliceLength * this.textureShape.depth;
        switch (this.textureShape.type) {
          case WebGL2RenderingContext.FLOAT: {
            const buffer = new Float32Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Float32Array', buffer };
          }
          case WebGL2RenderingContext.HALF_FLOAT: {
            const buffer = new Uint16Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Uint16Array', buffer };
          }
          case WebGL2RenderingContext.INT: {
            const buffer = new Int32Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Int32Array', buffer };
          }
          case WebGL2RenderingContext.UNSIGNED_BYTE: {
            const buffer = new Uint8Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Uint8Array', buffer };
          }
          default:
            throw new Error();
        }
      }
    }
  }

  setDataRaw(data: ArrayBufferView): void {
    const ctx = getNNWebGLContext();
    this.bindToReadTexture(0);
    switch (this.textureShape.dim) {
      case '2D':
        ctx.gl.texSubImage2D(
          this.target,
          0,
          0,
          0,
          this.textureShape.width,
          this.textureShape.height,
          this.textureShape.format,
          this.textureShape.type,
          data,
          0
        );
        break;
      case '2DArray':
        ctx.gl.texSubImage3D(
          this.target,
          0,
          0,
          0,
          0,
          this.textureShape.width,
          this.textureShape.height,
          this.textureShape.depth,
          this.textureShape.format,
          this.textureShape.type,
          data,
          0
        );
        break;
      default:
        throw new Error('not implemented');
    }
    this.unbindFromReadTexture();
  }

  private readPixels2D(buf: ArrayBufferView) {
    const ctx = getNNWebGLContext();
    this.bindToDrawTexture();
    ctx.gl.readPixels(
      0,
      0,
      this.textureShape.width,
      this.textureShape.height,
      this.textureShape.format,
      this.textureShape.type,
      buf
    );
    this.unbindFromDrawTexture();
  }

  private readPixels2DArray(
    buf: ArrayBufferView,
    sliceLength: number,
    depth: number
  ) {
    const ctx = getNNWebGLContext();
    for (let layer = 0; layer < depth; layer++) {
      this.bindToDrawTexture(layer);
      ctx.gl.readPixels(
        0,
        0,
        this.textureShape.width,
        this.textureShape.height,
        this.textureShape.format,
        this.textureShape.type,
        buf,
        sliceLength * layer
      );
      this.unbindFromDrawTexture();
    }
  }
}

export class WebGLTensor extends Tensor {
  buffer: WebGLTensorBuffer;

  private constructor(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: WebGLTensorBuffer,
    textureShape?: TensorTextureShape
  ) {
    super('webgl', shape, dtype);
    const ctx = getNNWebGLContext();
    if (buffer) {
      this.buffer = buffer;
    } else {
      this.buffer = new WebGLTensorBuffer(
        textureShape ||
          this.calcDefaultTextureShape(
            this.size,
            this.dtype,
            ctx.maxTextureSize,
            ctx.supportsTexture32bit
          )
      );
    }
  }

  private calcDefaultTextureShape(
    length: number,
    dtype: DType,
    maxTextureSize: number,
    supportsTexture32bit: boolean
  ): TensorTextureShape {
    const format = getTensorTextureShapeFormatForDType(
      dtype,
      supportsTexture32bit
    );
    if (length <= maxTextureSize) {
      return {
        dim: '2D',
        width: length,
        height: 1,
        ...format,
      };
    } else {
      let height = Math.ceil(length / maxTextureSize);
      if (height > maxTextureSize) {
        const depth = Math.ceil(height / maxTextureSize);
        height = maxTextureSize;
        return {
          dim: '2DArray',
          width: maxTextureSize,
          height: maxTextureSize,
          depth,
          ...format,
        };
      }
      return {
        dim: '2D',
        width: maxTextureSize,
        height,
        ...format,
      };
    }
  }

  alias(shape?: ArrayLike<number>): WebGLTensor {
    let t: WebGLTensor;
    const buffer = this.buffer;
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

  static s(value: number): WebGLTensor {
    return WebGLTensor.fromArray([value], []);
  }

  static empty(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault,
    buffer?: WebGLTensorBuffer,
    textureShape?: TensorTextureShape
  ): WebGLTensor {
    return new WebGLTensor(shape, dtype, buffer, textureShape);
  }

  static zeros(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): WebGLTensor {
    const data = new Float32Array(arrayProd(shape));
    return WebGLTensor.fromArray(data, shape, dtype);
  }

  static ones(
    shape: ArrayLike<number>,
    dtype: DType = DTypeDefault
  ): WebGLTensor {
    const data = new Float32Array(arrayProd(shape));
    data.fill(1);
    return WebGLTensor.fromArray(data, shape, dtype);
  }

  to(backend: 'cpu'): Promise<CPUTensor>;
  async to(backend: Backend): Promise<Tensor> {
    switch (backend) {
      case 'cpu':
        return CPUTensor.fromArray(
          await this.toTypedArrayAsync(),
          this.shape,
          this.dtype
        );
      case 'webgl':
        return this.alias();
      default:
        throw new Error(`Unknown backend ${backend}`);
    }
  }

  static fromArray(
    data: ArrayLike<number>,
    shape?: ArrayLike<number>,
    dtype?: DType
  ): WebGLTensor {
    const t = new WebGLTensor(shape || [data.length], dtype);
    t.setArray(data);
    return t;
  }

  setArray(data: ArrayLike<number>): void {
    if (data.length !== this.size) {
      throw new Error('length mismatch');
    }
    // pack to texture raw format
    let packed: ArrayBufferView;
    switch (this.buffer.textureShape.type) {
      case WebGL2RenderingContext.FLOAT:
        packed = packToFloat32Array(data, this.buffer.textureLength);
        break;
      case WebGL2RenderingContext.HALF_FLOAT:
        packed = packToFloat16Array(data, this.buffer.textureLength);
        break;
      case WebGL2RenderingContext.INT:
        packed = packToInt32Array(data, this.buffer.textureLength);
        break;
      case WebGL2RenderingContext.UNSIGNED_BYTE:
        packed = packToUint8Array(data, this.buffer.textureLength);
        break;
      default:
        throw new Error();
    }
    this.buffer.setDataRaw(packed);
  }

  async toTypedArrayAsync(): Promise<TypedArrayTypes> {
    // 同期的に行えるがブロックする。計算完了まで非同期的に待機することも考えられる。
    const rawData = this.buffer.getDataRaw();
    switch (rawData.type) {
      case 'Float32Array':
        return unpackFromFloat32Array(rawData.buffer, this.size);
      case 'Uint16Array':
        return unpackFromFloat16Array(rawData.buffer, this.size);
      case 'Int32Array':
        return unpackFromInt32Array(rawData.buffer, this.size);
      case 'Uint8Array':
        return unpackFromUint8Array(rawData.buffer, this.size);
      default:
        throw new Error();
    }
  }

  async toArrayAsync(): Promise<number[]> {
    return Array.from(await this.toTypedArrayAsync());
  }

  dispose() {
    if (!this.buffer) {
      throw new Error('Double-dispose of WebGLTensor');
    }
    this.buffer.ref--;
    if (this.buffer.ref <= 0) {
      this.buffer.dispose();
    }
    (this as { buffer: WebGLTensorBuffer | null }).buffer = null;
  }

  static add(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    return add(lhs, rhs);
  }

  static mul(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    // TODO: broadcast, type check
    if (!arrayEqual(lhs.shape, rhs.shape)) {
      throw new Error('shape not match');
    }
    throw new Error();
  }

  static exp(x: WebGLTensor): WebGLTensor {
    return exp(x);
  }
}

import { Backend } from '../../backend';
import { DType, DTypeDefault, TypedArrayTypes } from '../../dtype';
import { arrayProd } from '../../util';
import { CPUTensor } from '../cpu/cpuTensor';
import { calcReshape, calcTransposeShape } from '../shapeUtil';
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
import { broadcastTo, stridedCopy } from './core/copy';
import { gemm } from './core/gemm';
import {
  mseLoss,
  mseLossBackprop,
  nllLoss,
  softmax,
  softmaxCrossEntropyBackward,
} from './core/loss';
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
import { sum, sumTo } from './core/reduction';
import {
  getTypeForDType,
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  webglShaderHeader,
} from './core/shaderHelper';
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
} from './core/unary';
import { getNNWebGLContext } from './webglContext';

let webglAllocCount = 0;
export const existingBuffers: Set<WebGLTensorBuffer> = new Set();

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
  format: WebGL2RenderingContext.RGBA_INTEGER,
  type: WebGL2RenderingContext.INT,
};

export const tensorTextureShapeFormatRGBA8UI = {
  internalFormat: WebGL2RenderingContext.RGBA8UI,
  format: WebGL2RenderingContext.RGBA_INTEGER,
  type: WebGL2RenderingContext.UNSIGNED_BYTE,
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

export type TensorTextureShapeDim = '2D' | '2DArray';

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

export class WebGLTensorBuffer {
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
      case WebGL2RenderingContext.RGBA_INTEGER:
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

  private packRToRGBA(): WebGLTensorBuffer {
    if (this.dimPerPixel !== 1) {
      throw new Error('This buffer is RGBA');
    }

    const ctx = getNNWebGLContext();
    const dstPixels = Math.ceil(this.textureLength / 4);
    const width = ctx.maxTextureSize;
    const height = Math.ceil(dstPixels / width);
    if (height > ctx.maxTextureSize) {
      // 2DArrayへのコピーを実装すれば読み出し可能。未実装
      throw new Error(`Tensor too large`);
    }

    let dstShapeFormat: TensorTextureShapeFormat;

    let srcDtype: DType;
    let dstDtype: DType;
    switch (this.textureShape.internalFormat) {
      case WebGL2RenderingContext.R32F:
        dstShapeFormat = tensorTextureShapeFormatRGBA32F;
        srcDtype = dstDtype = 'float32';
        break;
      case WebGL2RenderingContext.R16F:
        dstShapeFormat = tensorTextureShapeFormatRGBA16F;
        srcDtype = dstDtype = 'float32';
        break;
      case WebGL2RenderingContext.R32I:
        dstShapeFormat = tensorTextureShapeFormatRGBA32I;
        srcDtype = dstDtype = 'int32';
        break;
      case WebGL2RenderingContext.R8UI:
        // Mac FirefoxでRGBA8UIが読み出せないためint32の状態で取り出す
        dstShapeFormat = tensorTextureShapeFormatRGBA32I;
        srcDtype = 'uint8';
        dstDtype = 'int32';
        break;
      default:
        throw new Error();
    }
    const { vec4Type, scalarType } = getTypeForDType(dstDtype);
    const dst = new WebGLTensorBuffer({
      ...dstShapeFormat,
      dim: '2D',
      width,
      height,
    });
    const kernelName = `packRToRGBA_${this.textureShape.dim}_${srcDtype}_${dstDtype}`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
${shaderGenTensorOutputUniform(1, dst.textureShape.dim, dstDtype)}
${shaderGenTensorNDGet('tex_input', 1, this.textureShape.dim, srcDtype)}
uniform int input_pixels;
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(1, dst.textureShape.dim)}
  ${vec4Type} v = ${vec4Type}(0.0, 0.0, 0.0, 0.0);
  int pos = tex_output_0 * 4;
  if (pos < input_pixels) {
    v.r = ${scalarType}(get_tex_input(pos));
  }
  pos++;
  if (pos < input_pixels) {
    v.g = ${scalarType}(get_tex_input(pos));
  }
  pos++;
  if (pos < input_pixels) {
    v.b = ${scalarType}(get_tex_input(pos));
  }
  pos++;
  if (pos < input_pixels) {
    v.a = ${scalarType}(get_tex_input(pos));
  }
  ${shaderGenOutput('v', srcDtype, true)};
}
      `
      );
    }
    ctx.runKernel(kernelName, [{ buffer: this, name: 'tex_input' }], dst, [
      { name: '_ka_tex_output_shape_0', type: 'int', value: dstPixels },
      { name: '_ka_tex_output_texture_h', type: 'int', value: height },
      { name: '_ka_tex_output_texture_w', type: 'int', value: width },
      { name: '_ka_tex_input_stride_0', type: 'int', value: 1 },
      { name: 'input_pixels', type: 'int', value: this.textureLength },
    ]);
    return dst;
  }

  getDataRaw():
    | { type: 'Float32Array'; buffer: Float32Array }
    | { type: 'Uint16Array'; buffer: Uint16Array }
    | { type: 'Int32Array'; buffer: Int32Array }
    | { type: 'Uint8Array'; buffer: Uint8Array } {
    const ctx = getNNWebGLContext();
    if (ctx.canOnlyReadRGBA && this.dimPerPixel === 1) {
      const packed = this.packRToRGBA();
      const packedData = packed.getDataRaw();
      packed.dispose();
      if (this.textureShape.internalFormat === WebGL2RenderingContext.R8UI) {
        // RGBA8UIが直接読めずInt32Arrayとして読まれるので、Uint8Arrayに変換してpackの有無での差異をなくす
        const uint8 = new Uint8Array(packedData.buffer.length);
        uint8.set(packedData.buffer);
        return { type: 'Uint8Array', buffer: uint8 };
      }
      return packedData;
    }
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
    // Mac + ChromeではRチャンネルのみのテクスチャを読み出せない
    // Mac + Firefoxではさらに、RGBA8UIも読み出せない
    // packRToRGBAで基本的に回避しているが、これを経由せずRGBA8UIを直接使うコードがあるとエラーになりうる
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

  static isWebGLTensor(tensor: unknown): tensor is WebGLTensor {
    return typeof tensor === 'object' && (tensor as Tensor).backend === 'webgl';
  }

  getClass(): typeof WebGLTensor {
    return WebGLTensor;
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
        width: Math.max(length, 1), // 最低でも1x1サイズを確保
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
  to(backend: Backend): Promise<Tensor>;
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
    return coreadd(lhs, rhs);
  }

  static sub(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    return coresub(lhs, rhs);
  }

  static mul(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    return coremul(lhs, rhs);
  }

  static div(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    return corediv(lhs, rhs);
  }

  static pow(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
    return corepow(lhs, rhs);
  }

  static abs(x: WebGLTensor): WebGLTensor {
    return coreabs(x);
  }

  static acos(x: WebGLTensor): WebGLTensor {
    return coreacos(x);
  }

  static acosh(x: WebGLTensor): WebGLTensor {
    return coreacosh(x);
  }

  static asin(x: WebGLTensor): WebGLTensor {
    return coreasin(x);
  }

  static asinh(x: WebGLTensor): WebGLTensor {
    return coreasinh(x);
  }

  static atan(x: WebGLTensor): WebGLTensor {
    return coreatan(x);
  }

  static atanh(x: WebGLTensor): WebGLTensor {
    return coreatanh(x);
  }

  static cos(x: WebGLTensor): WebGLTensor {
    return corecos(x);
  }

  static cosh(x: WebGLTensor): WebGLTensor {
    return corecosh(x);
  }

  static exp(x: WebGLTensor): WebGLTensor {
    return coreexp(x);
  }

  static log(x: WebGLTensor): WebGLTensor {
    return corelog(x);
  }

  static neg(x: WebGLTensor): WebGLTensor {
    return coreneg(x);
  }

  static relu(x: WebGLTensor): WebGLTensor {
    return corerelu(x);
  }

  static sigmoid(x: WebGLTensor): WebGLTensor {
    return coresigmoid(x);
  }

  static sin(x: WebGLTensor): WebGLTensor {
    return coresin(x);
  }

  static sinh(x: WebGLTensor): WebGLTensor {
    return coresinh(x);
  }

  static sqrt(x: WebGLTensor): WebGLTensor {
    return coresqrt(x);
  }

  static square(x: WebGLTensor): WebGLTensor {
    return coresquare(x);
  }

  static tan(x: WebGLTensor): WebGLTensor {
    return coretan(x);
  }

  static tanh(x: WebGLTensor): WebGLTensor {
    return coretanh(x);
  }

  /**
   * 転置込みの行列積を行う暫定的な関数
   * @param a
   * @param b
   * @param transa
   * @param transb
   */
  static gemm(
    a: WebGLTensor,
    b: WebGLTensor,
    transa = false,
    transb = false
  ): WebGLTensor {
    return gemm(a, b, transa, transb);
  }

  static dot(a: WebGLTensor, b: WebGLTensor): WebGLTensor {
    return WebGLTensor.gemm(a, b, false, false);
  }

  static broadcastTo(
    x: WebGLTensor,
    shape: ReadonlyArray<number>
  ): WebGLTensor {
    return broadcastTo(x, shape);
  }

  static sum(
    x: WebGLTensor,
    axis?: number | number[] | null,
    keepdims?: boolean
  ): WebGLTensor {
    return sum(x, axis, keepdims);
  }

  static sumTo(x: WebGLTensor, shape: ReadonlyArray<number>): WebGLTensor {
    return sumTo(x, shape);
  }

  static reshape(
    x: WebGLTensor,
    shape: ReadonlyArray<number> | number,
    allowZero = true
  ): WebGLTensor {
    return x.alias(calcReshape(x.shape, shape, allowZero));
  }

  static transpose(
    x: WebGLTensor,
    axes?: ReadonlyArray<number> | null
  ): WebGLTensor {
    const { newShape, srcStrides } = calcTransposeShape(
      x.shape,
      x.strides,
      axes
    );
    return stridedCopy(x, newShape, srcStrides);
  }

  static mseLossBackprop(
    ad: WebGLTensor,
    bd: WebGLTensor,
    gyd: WebGLTensor
  ): WebGLTensor[] {
    return mseLossBackprop(ad, bd, gyd);
  }

  static mseLoss(a: WebGLTensor, b: WebGLTensor): WebGLTensor {
    return mseLoss(a, b);
  }

  static nllLoss(softmax: WebGLTensor, label: WebGLTensor): WebGLTensor {
    return nllLoss(softmax, label);
  }

  static softmax(x: WebGLTensor): WebGLTensor {
    return softmax(x);
  }

  static softmaxCrossEntropyBackward(
    softmax: WebGLTensor,
    label: WebGLTensor,
    gy: WebGLTensor
  ): WebGLTensor {
    return softmaxCrossEntropyBackward(softmax, label, gy);
  }

  static reluBackprop(yd: WebGLTensor, gyd: WebGLTensor): WebGLTensor {
    return corereluBackprop(yd, gyd);
  }

  static sigmoidBackprop(yd: WebGLTensor, gyd: WebGLTensor): WebGLTensor {
    return coresigmoidBackprop(yd, gyd);
  }
}

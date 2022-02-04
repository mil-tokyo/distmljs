import { nonNull } from '../../util';
import {
  TensorTextureShape,
  WebGLTensor,
  WebGLTensorBuffer,
} from './webglTensor';

// [x y u v] * [upper-left, lower-left, upper-right, lower-right]
const vertexArray = new Float32Array([-1, +1, -1, -1, +1, +1, +1, -1]);
const vertex_shader_source_2 = `#version 300 es
precision highp float;
in vec2 _xy;
void main() { 
  gl_Position = vec4(_xy, 0, 1); 
}
`;

export interface WebGLUniformItem {
  name: string;
  value: number;
  type: 'float' | 'int';
}

// function deleteTextureWait() {
//   return new Promise<void>((resolve) => {
//     setTimeout(resolve, 1);
//   });
// }

function initWebGL() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 not supported');
  }
  const allowedTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  let maxTextureSize: number;
  if (allowedTextureSize >= 16384) {
    maxTextureSize = 16384;
  } else if (allowedTextureSize >= 4096) {
    maxTextureSize = 4096;
  } else {
    throw new Error(`gl.MAX_TEXTURE_SIZE is too small (${allowedTextureSize})`);
  }
  return { gl, maxTextureSize };
}

export interface WebGLKernelInputTensor {
  name: string;
  tensor: WebGLTensor;
}

export interface WebGLKernelInputBuffer {
  name: string;
  buffer: WebGLTensorBuffer;
}

export type WebGLKernelInput = WebGLKernelInputBuffer | WebGLKernelInputTensor;

export class NNWebGLContext {
  gl: WebGL2RenderingContext;
  maxTextureSize: number;
  fb: WebGLFramebuffer;
  supportsTexture32bit: boolean;
  supportsTexture16bit: boolean;
  canOnlyReadRGBA: boolean;
  private programs: Map<string, { program: WebGLProgram }> = new Map();
  private vshader!: WebGLShader;

  constructor() {
    const { gl, maxTextureSize } = initWebGL();
    this.gl = gl;
    this.maxTextureSize = maxTextureSize;

    if (gl.getExtension('EXT_color_buffer_float')) {
      // Enable color mode of gl.R32F
      this.supportsTexture32bit = true;
      // EXT_color_buffer_float が取得できればR16Fも含んでいる
      // これが取得できても、EXT_color_buffer_half_floatが取得できない環境もある
      this.supportsTexture16bit = true;
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      // Enable color mode of gl.R16F
      this.supportsTexture32bit = false;
      this.supportsTexture16bit = true;
    } else {
      // 浮動小数点数テクスチャが格納できない環境はサポート外
      throw new Error(
        'Neither EXT_color_buffer_float nor EXT_color_buffer_half_float are supported'
      );
    }
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.SAMPLE_COVERAGE);
    gl.enable(gl.SCISSOR_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    const vertexBuffer = this.createArrayBuffer(vertexArray);
    this.bindArrayBuffer(vertexBuffer);
    this.fb = nonNull(gl.createFramebuffer());
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);

    // バグ回避
    // Mac Chromeで、RチャンネルのみのテクスチャをreadPixelsで読みだそうとするとエラーとなる
    // GL ERROR :GL_INVALID_OPERATION : glReadPixels: format and type incompatible with the current read framebuffer
    const ua = navigator.userAgent;
    this.canOnlyReadRGBA = ua.includes('Macintosh') && ua.includes('Chrome/');
  }

  createArrayBuffer(vertexArray: Float32Array): WebGLBuffer {
    const buffer = nonNull(this.gl.createBuffer());
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, vertexArray, this.gl.STATIC_DRAW);

    return buffer;
  }

  bindArrayBuffer(buffer: WebGLBuffer): void {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
  }

  createTexture(textureShape: TensorTextureShape): WebGLTexture {
    if (
      textureShape.dim === '2DArray' &&
      (textureShape.width === 1 || textureShape.height === 1)
    ) {
      // texSubImage3D raises Error when the following condition is met
      // WebGL: INVALID_OPERATION: texSubImage3D: ArrayBufferView not big enough for request
      // (textureShape.dim === "2DArray" && (textureShape.width === 1 || textureShape.height === 1) && textureShape.internalFormat === WebGL2RenderingContext.R16F
      throw new Error(
        'The condition raises error: textureShape.dim === "2DArray" && (textureShape.width === 1 || textureShape.height === 1))'
      );
    }
    const gl = this.gl;
    const texture = nonNull(gl.createTexture());
    gl.activeTexture(gl.TEXTURE0);
    let target: number;
    switch (textureShape.dim) {
      case '2D':
        target = gl.TEXTURE_2D;
        gl.bindTexture(target, texture);
        gl.texStorage2D(
          target,
          1,
          textureShape.internalFormat,
          textureShape.width,
          textureShape.height
        );
        break;
      case '2DArray':
        target = gl.TEXTURE_2D_ARRAY;
        gl.bindTexture(target, texture);
        gl.texStorage3D(
          target,
          1,
          textureShape.internalFormat,
          textureShape.width,
          textureShape.height,
          textureShape.depth
        );
        break;
      default:
        throw new Error();
    }
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(target, null);

    return texture;
  }

  createShader(type: number, source: string): WebGLShader {
    const shader = nonNull(this.gl.createShader(type));

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw Error(`Shader Compile failed: ${this.gl.getShaderInfoLog(shader)}`);
    }

    return shader;
  }

  addKernel(name: string, sourceCode: string): void {
    if (this.programs.has(name)) {
      return;
    }
    this.programs.set(name, { program: this.compileKernel(sourceCode) });
  }

  hasKernel(name: string): boolean {
    return this.programs.has(name);
  }

  compileKernel(sourceCode: string): WebGLProgram {
    const { gl } = this;
    if (!this.vshader) {
      this.vshader = this.createShader(
        gl.VERTEX_SHADER,
        vertex_shader_source_2
      );
    }
    const fshader = this.createShader(gl.FRAGMENT_SHADER, sourceCode),
      program = nonNull(this.gl.createProgram());

    this.gl.attachShader(program, fshader);
    this.gl.attachShader(program, this.vshader);
    this.gl.linkProgram(program);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error('ShaderProgram Initialization failed.');
    }

    return program;
  }

  runKernel(
    name: string,
    inputs: WebGLKernelInput[],
    output: WebGLTensor | WebGLTensorBuffer,
    uniforms: WebGLUniformItem[],
    drawLayer: number | null = null
  ): void {
    let outputBuffer: WebGLTensorBuffer;
    if (output instanceof WebGLTensor) {
      outputBuffer = output.buffer;
    } else {
      outputBuffer = output;
    }
    if (outputBuffer.textureShape.dim === '2DArray' && drawLayer == null) {
      for (let d = 0; d < outputBuffer.textureShape.depth; d++) {
        this.runKernelSingleDrawLayer(name, inputs, outputBuffer, uniforms, d);
      }
    } else {
      this.runKernelSingleDrawLayer(
        name,
        inputs,
        outputBuffer,
        uniforms,
        drawLayer || 0
      );
    }
  }

  private runKernelSingleDrawLayer(
    name: string,
    inputs: WebGLKernelInput[],
    outputBuffer: WebGLTensorBuffer,
    uniforms: WebGLUniformItem[],
    drawLayer: number
  ): void {
    const { gl } = this;
    const kobj = this.programs.get(name);
    if (!kobj) {
      throw new Error(`Unknown kernel ${name}`);
    }

    const xyAttribLoc = gl.getAttribLocation(kobj.program, '_xy');
    for (let i = 0; i < inputs.length; i++) {
      const ini = inputs[i];
      let buffer: WebGLTensorBuffer;
      if ('tensor' in ini) {
        buffer = ini.tensor.buffer;
      } else {
        buffer = ini.buffer;
      }
      buffer.bindToReadTexture(i);
    }
    outputBuffer.bindToDrawTexture(drawLayer);

    gl.useProgram(kobj.program);

    const extendedUniforms: WebGLUniformItem[] = [
      ...uniforms,
      { type: 'int', name: '_ka_depth', value: drawLayer },
    ];
    for (let i = 0; i < inputs.length; i++) {
      extendedUniforms.push({ type: 'int', name: inputs[i].name, value: i });
    }

    for (const uniform of extendedUniforms) {
      const loc = gl.getUniformLocation(kobj.program, uniform.name);
      if (loc == null) {
        continue;
      }
      switch (uniform.type) {
        case 'float':
          gl.uniform1f(loc, uniform.value);
          break;
        case 'int':
          gl.uniform1i(loc, uniform.value);
          break;
        default:
          throw new Error();
      }
    }
    gl.vertexAttribPointer(xyAttribLoc, 2, gl.FLOAT, true, 8, 0);
    gl.enableVertexAttribArray(xyAttribLoc);

    const fbStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (fbStatus !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`FRAMEBUFFER status invalid: ${fbStatus}.`);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexArray.length / 2);
    // TODO: 完了を待つかどうか
    // gl.flush();
    // gl.finish();

    for (let i = 0; i < inputs.length; i++) {
      const ini = inputs[i];
      let buffer: WebGLTensorBuffer;
      if ('tensor' in ini) {
        buffer = ini.tensor.buffer;
      } else {
        buffer = ini.buffer;
      }
      buffer.unbindFromReadTexture();
    }

    outputBuffer.unbindFromDrawTexture();
  }
}

let context: NNWebGLContext | null = null;
export async function initializeNNWebGLContext(): Promise<void> {
  // 現状非同期処理はないが、将来的に機能テストなどを加える可能性がある
  context = new NNWebGLContext();
}

export function getNNWebGLContext(): NNWebGLContext {
  if (!context) {
    throw new Error('WebGL Context does not exist');
  }
  return context;
}

import { nonNull } from '../../util';

// [x y u v] * [upper-left, lower-left, upper-right, lower-right]
const vertexArray = new Float32Array([-1, +1, -1, -1, +1, +1, +1, -1]),
  vertex_shader_source_1 = `
precision highp float;
attribute vec2 _xy;
void main() { 
  gl_Position = vec4(_xy, 0, 1); 
}
`,
  vertex_shader_source_2 = `#version 300 es
precision highp float;
in vec2 _xy;
void main() { 
  gl_Position = vec4(_xy, 0, 1); 
}
`;

function deleteTextureWait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 1);
  });
}

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

export class NNWebGLContext {
  gl: WebGL2RenderingContext;
  maxTextureSize: number;
  fb: WebGLFramebuffer;

  constructor() {
    const { gl, maxTextureSize } = initWebGL();
    this.gl = gl;
    this.maxTextureSize = maxTextureSize;
    // Enable color mode of gl.R32F
    gl.getExtension('EXT_color_buffer_float');
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
}

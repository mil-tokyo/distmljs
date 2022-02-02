import { getNNWebGLContext, webglShaderHeader } from '../webglContext';
import { WebGLTensor } from '../webglTensor';

export function exp(x: WebGLTensor): WebGLTensor {
  if (x.dtype !== 'float32') {
    throw new Error();
  }
  if (x.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error();
  }
  // TODO: unaryに一般化
  const output = WebGLTensor.empty(
    x.shape,
    x.dtype,
    undefined,
    x.buffer.textureShape
  );
  const ctx = getNNWebGLContext();
  if (output.buffer.textureShape.dim === '2D') {
    if (!ctx.hasKernel('exp')) {
      ctx.addKernel(
        'exp',
        webglShaderHeader +
          `
uniform sampler2D tex_input;
void main() {
  float r = texelFetch(tex_input, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0).r;
  fragColor = vec4(exp(r), 0.0, 0.0, 0.0);
}
`
      );
    }
    ctx.runKernel('exp', [{ tensor: x, name: 'tex_input' }], output, []);
  } else {
    if (!ctx.hasKernel('exp2d')) {
      ctx.addKernel(
        'exp2d',
        webglShaderHeader +
          `
uniform sampler2DArray tex_input;
uniform int _ka_depth;
out vec4 fragColor;
void main() {
  float r = texelFetch(tex_input, ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), _ka_depth), 0).r;
  fragColor = vec4(exp(r), 0.0, 0.0, 0.0);
}
`
      );
    }
    ctx.runKernel('exp2d', [{ tensor: x, name: 'tex_input' }], output, []);
  }
  return output;
}

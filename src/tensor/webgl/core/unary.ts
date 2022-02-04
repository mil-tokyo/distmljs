import { DType } from '../../../dtype';
import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';
import {
  shaderGenOutput,
  shaderGenTensorElementwiseGet,
  shaderGenTensorOutputUniformElementwise,
  shaderGenTensorOutputUniformElementwiseItem,
  webglShaderHeader,
} from './shaderHelper';

function unaryWrap(
  x: WebGLTensor,
  name: string,
  exprs: { [T in DType]?: string }
): WebGLTensor {
  if (x.buffer.dimPerPixel !== 1) {
    throw new Error(`${name}: RGBA texture is not yet supported`);
  }
  const dtype = x.dtype;
  const dim = x.buffer.textureShape.dim;
  let returnType: string;
  const expr = exprs[dtype];
  if (!expr) {
    throw new Error(`${name}: dtype ${dtype} is not supported`);
  }
  switch (dtype) {
    case 'float32':
      returnType = 'float';
      break;
    case 'int32':
      returnType = 'int';
      break;
    default:
      returnType = 'uint';
      break;
  }
  const output = WebGLTensor.empty(
    x.shape,
    x.dtype,
    undefined,
    x.buffer.textureShape
  );
  const ctx = getNNWebGLContext();
  // TODO: カーネル名マングリング手段の一般化
  const kernelName = `unary_${name}_${dim}_${dtype}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
${shaderGenTensorOutputUniformElementwise(dim, dtype)}
${shaderGenTensorElementwiseGet('tex_input', dim, dtype)}
void main() {
  ${returnType} v_s = get_tex_input();
  ${expr}
  ${shaderGenOutput('v', dtype)}
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_input' }], output, [
    ...shaderGenTensorOutputUniformElementwiseItem(output),
  ]);
  return output;
}

export function coreexp(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'exp', { float32: 'float v = exp(v_s);' });
}

export function coreabs(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'abs', {
    float32: 'float v = abs(v_s);',
    int32: 'int v = abs(v_s);',
  });
}

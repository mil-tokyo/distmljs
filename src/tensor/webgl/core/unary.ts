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

export function coreabs(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'abs', {
    float32: 'float v = abs(v_s);',
    int32: 'int v = abs(v_s);',
  });
}

export function coreacos(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'acos', { float32: 'float v = acos(v_s);' });
}

export function coreacosh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'acosh', { float32: 'float v = acosh(v_s);' });
}

export function coreasin(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'asin', { float32: 'float v = asin(v_s);' });
}

export function coreasinh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'asinh', { float32: 'float v = asinh(v_s);' });
}

export function coreatan(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'atan', { float32: 'float v = atan(v_s);' });
}

export function coreatanh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'atanh', { float32: 'float v = atanh(v_s);' });
}

export function corecopy(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'copy', {
    float32: 'float v = v_s;',
    int32: 'int v = v_s;',
    uint8: 'uint v = v_s;',
    bool: 'uint v = v_s;',
  });
}

export function corecos(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'cos', { float32: 'float v = cos(v_s);' });
}

export function corecosh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'cosh', { float32: 'float v = cosh(v_s);' });
}

export function coreexp(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'exp', { float32: 'float v = exp(v_s);' });
}

export function corelog(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'log', { float32: 'float v = log(v_s);' });
}

export function coreneg(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'neg', {
    float32: 'float v = -v_s;',
    int32: 'int v = -v_s;',
  });
}

export function corerelu(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'relu', { float32: 'float v = max(v_s, 0.0);' });
}

export function coresigmoid(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'sigmoid', {
    float32: 'float v = 1.0 / (1.0 + exp(-v_s));',
  });
}

export function coresin(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'sin', { float32: 'float v = sin(v_s);' });
}

export function coresinh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'sinh', { float32: 'float v = sinh(v_s);' });
}

export function coresqrt(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'sqrt', { float32: 'float v = sqrt(v_s);' });
}

export function coresquare(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'square', { float32: 'float v = v_s * v_s;' });
}

export function coretan(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'tan', { float32: 'float v = tan(v_s);' });
}

export function coretanh(x: WebGLTensor): WebGLTensor {
  return unaryWrap(x, 'tanh', { float32: 'float v = tanh(v_s);' });
}

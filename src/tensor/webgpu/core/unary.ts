import { DType } from '../../../dtype';
import { webgpuShaders } from '../shaders';
import { getNNWebGPUContext } from '../webgpuContext';
import { WebGPUTensor } from '../webgpuTensor';

function unaryWrap(x: WebGPUTensor, name: string): WebGPUTensor {
  const dtype = x.dtype;
  const shaderName = `unary_${name}_${dtype}`;
  const ctx = getNNWebGPUContext();
  if (!ctx.hasPipeline(shaderName)) {
    const shader = webgpuShaders[shaderName];
    if (!shader) {
      throw new Error(`${name}: dtype ${dtype} is not supported`);
    }
    ctx.createPipeline(shaderName, shader, 3);
  }
  const y = WebGPUTensor.empty(x.shape, x.dtype);
  ctx.runKernel({
    pipelineName: shaderName,
    tensors: [x, y],
    meta: {
      elements: [{ value: x.size, type: 'uint32' }],
    },
    workGroups: { x: 4096 / 64, y: 1, z: 1 },
  });

  return y;
}

export function coreabs(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'abs');
}

export function coreacos(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'acos');
}

export function coreacosh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'acosh');
}

export function coreasin(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'asin');
}

export function coreasinh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'asinh');
}

export function coreatan(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'atan');
}

export function coreatanh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'atanh');
}

export function corecopy(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'copy');
}

export function corecos(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'cos');
}

export function corecosh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'cosh');
}

export function coreexp(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'exp');
}

export function corelog(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'log');
}

export function coreneg(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'neg');
}

export function corerelu(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'relu');
}

export function coresigmoid(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'sigmoid');
}

export function coresin(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'sin');
}

export function coresinh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'sinh');
}

export function coresqrt(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'sqrt');
}

export function coresquare(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'square');
}

export function coretan(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'tan');
}

export function coretanh(x: WebGPUTensor): WebGPUTensor {
  return unaryWrap(x, 'tanh');
}

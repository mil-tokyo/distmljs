import { CPUTensor, Tensor, WebGLTensor, WebGPUTensor } from '.';
import { TensorStatic } from './tensor';

export function isAllCPUTensor(tensors: unknown[]): tensors is CPUTensor[] {
  for (const tensor of tensors) {
    if (!CPUTensor.isCPUTensor(tensor)) {
      return false;
    }
  }
  return true;
}

export function isAllWebGLTensor(tensors: unknown[]): tensors is WebGLTensor[] {
  for (const tensor of tensors) {
    if (!WebGLTensor.isWebGLTensor(tensor)) {
      return false;
    }
  }
  return true;
}

export function isAllWebGPUTensor(
  tensors: unknown[]
): tensors is WebGPUTensor[] {
  for (const tensor of tensors) {
    if (!WebGPUTensor.isWebGPUTensor(tensor)) {
      return false;
    }
  }
  return true;
}

export function getBackendOfTensors(
  tensors: unknown[]
):
  | { c: typeof CPUTensor; t: CPUTensor[] }
  | { c: typeof WebGLTensor; t: WebGLTensor[] }
  | { c: typeof WebGPUTensor; t: WebGPUTensor[] } {
  if (tensors.length === 0) {
    return { c: CPUTensor, t: [] };
  }
  if (tensors[0] instanceof CPUTensor) {
    if (!isAllCPUTensor(tensors)) {
      throw new Error('Tensor type not unified');
    }
    return { c: CPUTensor, t: tensors };
  } else if (tensors[0] instanceof WebGLTensor) {
    if (!isAllWebGLTensor(tensors)) {
      throw new Error('Tensor type not unified');
    }
    return { c: WebGLTensor, t: tensors };
  } else if (tensors[0] instanceof WebGPUTensor) {
    if (!isAllWebGPUTensor(tensors)) {
      throw new Error('Tensor type not unified');
    }
    return { c: WebGPUTensor, t: tensors };
  }
  throw new Error('Tensor type not unified');
}

export function genCall(
  tensors: unknown[],
  fs: {
    all?: <B extends CPUTensor | WebGLTensor | WebGPUTensor>(
      T: TensorStatic<B>,
      tensors: B[]
    ) => B[];
    cpu?: (c: typeof CPUTensor, tensors: CPUTensor[]) => CPUTensor[];
    webgl?: (c: typeof WebGLTensor, tensors: WebGLTensor[]) => WebGLTensor[];
    webgpu?: (
      c: typeof WebGPUTensor,
      tensors: WebGPUTensor[]
    ) => WebGPUTensor[];
  }
): Tensor[] {
  if (isAllCPUTensor(tensors)) {
    if (fs.cpu) {
      return fs.cpu(CPUTensor, tensors);
    } else if (fs.all) {
      return fs.all(CPUTensor, tensors);
    } else {
      throw new Error('Operation for cpu is not implemented');
    }
  } else if (isAllWebGLTensor(tensors)) {
    if (fs.webgl) {
      return fs.webgl(WebGLTensor, tensors);
    } else if (fs.all) {
      return fs.all(WebGLTensor, tensors);
    } else {
      throw new Error('Operation for webgl is not implemented');
    }
  } else if (isAllWebGPUTensor(tensors)) {
    if (fs.webgpu) {
      return fs.webgpu(WebGPUTensor, tensors);
    } else if (fs.all) {
      return fs.all(WebGPUTensor, tensors);
    } else {
      throw new Error('Operation for webgpu is not implemented');
    }
  } else {
    throw new Error('Tensor type not unified');
  }
}

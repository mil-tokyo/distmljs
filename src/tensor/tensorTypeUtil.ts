import { CPUTensor, Tensor, WebGLTensor } from '.';

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

export function getBackendOfTensors(
  tensors: unknown[]
):
  | { c: typeof CPUTensor; t: CPUTensor[] }
  | { c: typeof WebGLTensor; t: WebGLTensor[] } {
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
  }
  throw new Error('Tensor type not unified');
}

export function genCall(
  tensors: unknown[],
  fs: {
    cpu: (c: typeof CPUTensor, tensors: CPUTensor[]) => CPUTensor[];
    webgl: (c: typeof WebGLTensor, tensors: WebGLTensor[]) => WebGLTensor[];
  }
): Tensor[] {
  // TODO: どのバックエンドでも1つ関数を書けばいいような型チェック手段
  // genCall(tensors, (c,t) => [c.add(t[0])])
  // のようにしたい(型チェックを無視すればできるのだが)
  if (isAllCPUTensor(tensors)) {
    return fs.cpu(CPUTensor, tensors);
  } else if (isAllWebGLTensor(tensors)) {
    return fs.webgl(WebGLTensor, tensors);
  } else {
    throw new Error('Tensor type not unified');
  }
}

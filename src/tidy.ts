import { Variable } from './nn';
import { Layer, Optimizer } from './nn/core';
import { Tensor } from './tensor';
import {
  existingBuffers as existingWebGLBuffers,
  WebGLTensor,
} from './tensor/webgl/webglTensor';
import {
  existingBuffers as existingWebGPUBuffers,
  WebGPUTensor,
} from './tensor/webgpu/webgpuTensor';

type TensorKind = Tensor | Variable | Layer | Optimizer;
type TidyResult = TensorKind[];
export function tidy<T extends TidyResult>(fn: () => Promise<T>): Promise<T>;
export function tidy<T extends TidyResult>(
  name: string,
  fn: () => Promise<T>
): Promise<T>;
export async function tidy<T extends TidyResult>(
  nameOrFn: string | (() => Promise<T>),
  fn?: () => Promise<T>
): Promise<T> {
  let fn_: () => Promise<T>;
  let name: string | null;
  if (fn) {
    fn_ = fn;
    name = nameOrFn as string;
  } else {
    fn_ = nameOrFn as () => Promise<T>;
    name = null;
  }
  const keepWebGLBuffers = new Set(existingWebGLBuffers);
  const keepWebGPUBuffers = new Set(existingWebGPUBuffers);
  const returned = await fn_();
  const lastWebGLBuffers = new Set(existingWebGLBuffers);
  const lastWebGPUBuffers = new Set(existingWebGPUBuffers);
  // lastBuffers - firstBuffers - returned を開放
  const returnedTensors: Tensor[] = [];
  for (const r of returned) {
    if (r instanceof Tensor) {
      returnedTensors.push(r);
    } else if (r instanceof Variable) {
      returnedTensors.push(r.data);
    } else if (r instanceof Layer) {
      for (const param of r.parameters(true, false)) {
        returnedTensors.push(param.data);
      }
    } else if (r instanceof Optimizer) {
      returnedTensors.push(...r.getKeepTensors());
    }
  }
  for (const r of returnedTensors) {
    if (WebGLTensor.isWebGLTensor(r)) {
      keepWebGLBuffers.add(r.buffer);
    }
    if (WebGPUTensor.isWebGPUTensor(r)) {
      keepWebGPUBuffers.add(r.buffer);
    }
  }
  for (const fb of keepWebGLBuffers) {
    lastWebGLBuffers.delete(fb);
  }
  for (const fb of keepWebGPUBuffers) {
    lastWebGPUBuffers.delete(fb);
  }
  for (const buf of lastWebGLBuffers) {
    buf.dispose();
  }
  for (const buf of lastWebGPUBuffers) {
    buf.dispose();
  }

  return returned;
}
export function tidySync<T extends TidyResult>(fn: () => T): T;
export function tidySync<T extends TidyResult>(name: string, fn: () => T): T;
export function tidySync<T extends TidyResult>(
  nameOrFn: string | (() => T),
  fn?: () => T
): T {
  let fn_: () => T;
  let name: string | null;
  if (fn) {
    fn_ = fn;
    name = nameOrFn as string;
  } else {
    fn_ = nameOrFn as () => T;
    name = null;
  }
  const keepWebGLBuffers = new Set(existingWebGLBuffers);
  const keepWebGPUBuffers = new Set(existingWebGPUBuffers);
  const returned = fn_();
  const lastWebGLBuffers = new Set(existingWebGLBuffers);
  const lastWebGPUBuffers = new Set(existingWebGPUBuffers);
  // lastBuffers - firstBuffers - returned を開放
  const returnedTensors: Tensor[] = [];
  for (const r of returned) {
    if (r instanceof Tensor) {
      returnedTensors.push(r);
    } else if (r instanceof Variable) {
      returnedTensors.push(r.data);
    } else if (r instanceof Layer) {
      for (const param of r.parameters(true, false)) {
        returnedTensors.push(param.data);
      }
    } else if (r instanceof Optimizer) {
      returnedTensors.push(...r.getKeepTensors());
    }
  }
  for (const r of returnedTensors) {
    if (WebGLTensor.isWebGLTensor(r)) {
      keepWebGLBuffers.add(r.buffer);
    }
    if (WebGPUTensor.isWebGPUTensor(r)) {
      keepWebGPUBuffers.add(r.buffer);
    }
  }
  for (const fb of keepWebGLBuffers) {
    lastWebGLBuffers.delete(fb);
  }
  for (const fb of keepWebGPUBuffers) {
    lastWebGPUBuffers.delete(fb);
  }
  for (const buf of lastWebGLBuffers) {
    buf.dispose();
  }
  for (const buf of lastWebGPUBuffers) {
    buf.dispose();
  }

  return returned;
}

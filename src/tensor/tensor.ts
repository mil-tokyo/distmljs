import { CPUTensor } from './cpu/cpuTensor';
import { Backend } from '../backend';
import { DType } from '../dtype';
import { WebGLTensor } from '.';

export abstract class Tensor {
  readonly backend: Backend;
  readonly dtype: DType;
  readonly shape: ReadonlyArray<number>;
  readonly strides: ReadonlyArray<number>;
  readonly ndim: number;
  readonly size: number;

  constructor(backend: Backend, shape: ArrayLike<number>, dtype: DType) {
    this.backend = backend;
    this.dtype = dtype;
    this.shape = Array.from(shape).map((v) => Number(v) | 0);
    this.ndim = this.shape.length;
    const strides: number[] = [];
    let size = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      strides.unshift(size);
      size *= this.shape[i];
    }
    this.strides = strides;
    this.size = size;
  }

  abstract alias(shape?: ArrayLike<number>): Tensor;
  abstract to(backend: 'cpu'): Promise<CPUTensor>;
  abstract to(backend: Backend): Promise<Tensor>;
  abstract getClass(): typeof CPUTensor | typeof WebGLTensor;

  abstract dispose(): void;
}

import { Tensor } from './tensor';

export class CPUTensor extends Tensor {
  data: Float32Array;

  constructor(shape: ArrayLike<number>) {
    super(shape);
    this.data = new Float32Array(this.size);
  }
}

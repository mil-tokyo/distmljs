export abstract class Tensor {
  readonly shape: ReadonlyArray<number>;
  readonly ndim: number;
  readonly size: number;

  constructor(shape: ArrayLike<number>) {
    this.shape = Array.from(shape);
    this.ndim = this.shape.length;
    this.size = this.shape.reduce((c, v) => c * v, 1);
  }
}

/**
 * Means slice (e.g."array[2:5]") as in numpy. Use `slice` function to construct.
 */
export class Slice {
  constructor(
    public start: number | null = null,
    public stop: number | null = null,
    public step: number | null = null
  ) {}

  toString() {
    const start_ = this.start == null ? '' : this.start.toString();
    const stop_ = this.stop == null ? '' : this.stop.toString();
    const step_ = this.step == null ? '' : this.step.toString();
    if (step_) {
      return `${start_}:${stop_}:${step_}`;
    } else {
      return `${start_}:${stop_}`;
    }
  }
}

/**
 * Creates slice object used in {@link tensor.CPUTensor.gets}
 * @param start
 * @param stop
 * @param step
 * @returns slice object.
 */
export function slice(
  start: number | null = null,
  stop: number | null = null,
  step: number | null = null
): Slice {
  return new Slice(start, stop, step);
}

/**
 * Means "..." as in numpy. Use `ellipsis` constant.
 */
export class Ellipsis {
  toString() {
    return '...';
  }
}

/**
 * Means "..." as in numpy. Used in CPUTensor.gets().
 */
export const ellipsis = new Ellipsis();

/**
 * Means np.newaxis as in numpy. Used in CPUTensor.gets().
 */
export const newaxis = null;

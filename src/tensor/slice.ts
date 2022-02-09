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

export function slice(
  start: number | null = null,
  stop: number | null = null,
  step: number | null = null
): Slice {
  return new Slice(start, stop, step);
}

export class Ellipsis {
  toString() {
    return '...';
  }
}

export const ellipsis = new Ellipsis();

export const newaxis = null;

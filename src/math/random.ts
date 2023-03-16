/**
 * Random number / vector generator.
 */
export class Random {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(seed = 0) {
    // Algorithm: XorShift
    this.x = seed | 0;
    this.y = 362436069;
    this.z = 521288629;
    this.w = 88675123;
    // skip some initial values to decorrelate seed
    for (let i = 0; i < 40; i++) {
      this.randomRaw();
    }
  }

  static getDefault(): Random {
    return defaultInstance;
  }

  /**
   * Generates random integer
   * @returns Random integer [-2**31, 2**31-1]
   */
  randomRaw(): number {
    const x = this.x;
    const t = x ^ (x << 11);
    this.x = this.y;
    this.y = this.z;
    const w = this.w;
    this.z = w;
    const nw = w ^ (w >>> 19) ^ (t ^ (t >>> 8));
    this.w = nw;
    return nw; // 32bit signed integer
  }

  /**
   * Generates random number between [0, 1)
   * @param size spceify number to specify vector length
   */
  random(size?: null): number;
  random(size: number): Float32Array;
  random(size?: null | number): number | Float32Array {
    if (size == null) {
      // scalar number
      let raw = this.randomRaw(); // [-2**31, 2**31-1]
      raw += 2147483648; // [0, 2**32-1]
      return raw / 4294967296; // [0, 1)
    } else {
      // Float32Array
      const v = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        let raw = this.randomRaw(); // [-2**31, 2**31-1]
        raw += 2147483648; // [0, 2**32-1]
        const s = raw / 4294967296; // [0, 1)
        v[i] = s;
      }
      return v;
    }
  }
  /**
   * Generates random number between [range.low, range.high)
   * @param size spceify number to specify vector length
   */
  uniform(range: { low: number; high: number }, size?: null): number;
  uniform(range: { low: number; high: number }, size: number): Float32Array;
  uniform(
    range: { low: number; high: number },
    size?: null | number
  ): number | Float32Array {
    const low = range.low;
    const scale = range.high - range.low;
    if (size == null) {
      // scalar number
      let raw = this.randomRaw(); // [-2**31, 2**31-1]
      raw += 2147483648; // [0, 2**32-1]
      return (raw / 4294967296) * scale + low; // [0, 1) * scale + row
    } else {
      // Float32Array
      const v = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        let raw = this.randomRaw(); // [-2**31, 2**31-1]
        raw += 2147483648; // [0, 2**32-1]
        const s = raw / 4294967296; // [0, 1)
        v[i] = s * scale + low;
      }
      return v;
    }
  }
  /**
   * Generates random number from normal distribution.
   * @param size spceify number to specify vector length
   */
  normal(size?: null): number;
  normal(size: number): Float32Array;
  normal(size?: null | number): number | Float32Array {
    // Box–Muller's method
    if (size == null) {
      // scalar number
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const x = this.random();
        const alpha = Math.sqrt(-2 * Math.log(x));
        if (!Number.isFinite(alpha)) {
          // very rare case
          continue;
        }
        const y = this.random();
        const z1 = alpha * Math.cos(Math.PI * 2 * y);
        return z1;
      }
    } else {
      // Float32Array
      const v = new Float32Array(size);
      for (let i = 0; i < size; i += 2) {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const x = this.random();
          const alpha = Math.sqrt(-2 * Math.log(x));
          if (!Number.isFinite(alpha)) {
            // very rare case
            continue;
          }
          const y = this.random();
          const z1 = alpha * Math.cos(Math.PI * 2 * y);
          const z2 = alpha * Math.sin(Math.PI * 2 * y);
          v[i] = z1;
          if (i + 1 < size) {
            v[i + 1] = z2;
          }
          break;
        }
      }
      return v;
    }
  }

  private randIntScalar(max: number): number {
    const ceil = Math.floor(4294967296 / max) * max;
    let v: number;
    do {
      v = this.randomRaw() >>> 0; // to unsigned int
    } while (v >= ceil);
    return v % max;
  }

  private randintArray(max: number, size: number): Int32Array {
    const ret = new Int32Array(size);
    const ceil = Math.floor(4294967296 / max) * max;
    for (let i = 0; i < size; i++) {
      let v: number;
      do {
        v = this.randomRaw() >>> 0; // to unsigned int
      } while (v >= ceil);
      ret[i] = v % max;
    }

    return ret;
  }

  /**
   * Generates random integer between [range.low, range.high)
   * @param size spceify number to specify vector length
   */
  randint(range: { low: number; high: number }, size?: null): number;
  randint(range: { low: number; high: number }, size: number): Int32Array;
  randint(
    range: { low: number; high: number },
    size?: null | number
  ): number | Int32Array {
    const low = range.low;
    const scale = range.high - range.low;
    if (size == null) {
      // scalar number
      const raw = this.randIntScalar(scale);
      return raw + low;
    } else {
      // Int32Array
      const v = this.randintArray(scale, size);
      if (low !== 0) {
        for (let i = 0; i < size; i++) {
          v[i] += low;
        }
      }
      return v;
    }
  }

  randperm(size: number): Int32Array {
    const v = new Int32Array(size);
    for (let i = 0; i < size; i++) {
      v[i] = i;
    }

    // Fisher-Yates shuffle
    for (let i = 0; i < size - 1; i++) {
      const j = i + this.randIntScalar(size - i);
      const tmp = v[j];
      v[j] = v[i];
      v[i] = tmp;
    }

    return v;
  }
}

const defaultInstance = new Random();

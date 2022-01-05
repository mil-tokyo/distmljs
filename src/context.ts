// TODO: タスクごとに別のcontextを使用できるようにする

export interface NNContextValue {
  enableBackprop: boolean;
  train: boolean;
}

const NNContextValueDefault: NNContextValue = {
  enableBackprop: true,
  train: true,
};

export class NNContext {
  private value: NNContextValue;
  constructor(value = NNContextValueDefault) {
    this.value = { ...value };
  }

  get<K extends keyof NNContextValue>(key: K): NNContextValue[K] {
    return this.value[key];
  }

  set<K extends keyof NNContextValue>(key: K, value: NNContextValue[K]): void {
    this.value[key] = value;
  }

  async withValue<K extends keyof NNContextValue>(
    key: K,
    value: NNContextValue[K],
    fn: () => Promise<unknown>
  ): Promise<void> {
    const lastValue = this.value[key];
    try {
      this.value[key] = value;
      await fn();
    } finally {
      this.value[key] = lastValue;
    }
  }
}

export const defaultNNContext = new NNContext();

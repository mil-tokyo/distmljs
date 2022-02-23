import { Layer, Variable } from '../core';

export class Sequential extends Layer {
  [index: number]: Layer;
  length: number;

  constructor(...layers: Layer[]) {
    super();
    for (let i = 0; i < layers.length; i++) {
      this[i] = layers[i];
    }
    this.length = layers.length;
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let vs = inputs;
    for (let i = 0; i < this.length; i++) {
      vs = await this[i].call(...vs);
    }
    return vs;
  }
}

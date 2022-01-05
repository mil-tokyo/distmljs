import { defaultNNContext } from '../context';
import { CPUTensor } from '../tensor/cpuTensor';
import { Tensor } from '../tensor/tensor';
import { arrayEqual, nonNull } from '../util';

export class Variable {
  // TODO: requies_grad
  // TODO: detach(): Variableをコピーしてchainを切る
  grad?: Variable;
  creator?: NNFunction;
  generation: number;
  constructor(public data: Tensor, public name?: string) {
    this.generation = 0;
  }

  setCreator(func: NNFunction): void {
    this.creator = func;
    if (func.generation == null) {
      throw new Error('');
    }
    this.generation = func.generation + 1;
  }

  unchain(): void {
    this.creator = undefined;
  }

  cleargrad(): void {
    this.grad = undefined;
  }

  async backward(retainGrad = false, createGraph = false): Promise<void> {
    if (!this.grad) {
      const t = CPUTensor.ones(this.data.shape);
      this.grad = new Variable(t);
    }

    const funcs: NNFunction[] = [];
    const seenSet: Set<NNFunction> = new Set();

    const addFunc = (f: NNFunction): void => {
      if (!seenSet.has(f)) {
        funcs.push(f);
        seenSet.add(f);
        funcs.sort((a, b) => (a.generation || 0) - (b.generation || 0));
      }
    };

    if (!this.creator) {
      throw new Error();
    }
    addFunc(this.creator);

    while (funcs.length > 0) {
      const f = funcs.pop();
      if (!f?.outputs) {
        throw new Error();
      }
      const gys: Variable[] = f.outputs.map((wv) => nonNull(wv.deref()?.grad));

      await defaultNNContext.withValue(
        'enableBackprop',
        createGraph,
        async () => {
          const gxs = await f.backward(gys);
          for (let i = 0; i < gxs.length; i++) {
            const x = nonNull(f.inputs)[i];
            const gx = gxs[i];
            if (gx) {
              if (x.grad) {
                x.grad = (await new Add().call(x.grad, gx))[0];
              } else {
                x.grad = gx;
              }
              if (x.creator) {
                addFunc(x.creator);
              }
            }
          }
        }
      );

      if (!retainGrad) {
        f.outputs?.forEach((wv) => {
          const v = wv.deref();
          if (v) {
            // TODO: free
            v.grad = undefined;
          }
        });
      }
    }
  }

  unchainBackward() {
    if (this.creator) {
      const funcs = [this.creator];
      while (funcs.length > 0) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const f = funcs.pop()!;
        f.inputs?.forEach((v) => {
          if (v.creator) {
            funcs.push(v.creator);
            v.unchain();
          }
        });
      }
    }
  }
}

export class Parameter extends Variable {}

export abstract class NNFunction {
  generation?: number;
  inputs?: Variable[];
  outputs?: WeakRef<Variable>[];

  async call(...inputs: Variable[]): Promise<Variable[]> {
    const ys = await this.forward(inputs.map((v) => v.data));
    const outputs = ys.map((t) => new Variable(t));

    if (defaultNNContext.get('enableBackprop')) {
      this.inputs = inputs;
      this.generation = Math.max(...inputs.map((v) => v.generation));
      outputs.forEach((v) => v.setCreator(this));
      this.outputs = outputs.map((v) => new WeakRef(v));
    }

    return outputs;
  }

  /**
   * Syntax suger for call() with only one output variable.
   * @param inputs Input variables.
   * @returns Promise of output variable.
   */
  async c(...inputs: Variable[]): Promise<Variable> {
    const outputs = await this.call(...inputs);
    if (outputs.length !== 1) {
      throw new Error(
        'c() can only be used for NNFunction with one output variable.'
      );
    }
    return outputs[0];
  }

  abstract forward(inputs: Tensor[]): Promise<Tensor[]>;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  backward(inputs: Variable[]): Promise<(Variable | null)[]> {
    throw new Error('Not implemented');
  }
}

export class BroadcastTo extends NNFunction {
  constructor(public shape: ReadonlyArray<number>) {
    super();
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.broadcastTo(x as CPUTensor, this.shape)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new SumTo(this.inputs![0].data.shape).c(gy)];
  }
}

export class SumTo extends NNFunction {
  constructor(public shape: ReadonlyArray<number>) {
    super();
  }

  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.sumTo(x as CPUTensor, this.shape)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new BroadcastTo(this.inputs![0].data.shape).c(gy)];
  }
}

export class Sum extends NNFunction {
  constructor(
    public axis?: number | number[] | null,
    public keepdims?: boolean
  ) {
    super();
  }
  async forward([x]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.sum(x as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new BroadcastTo(this.inputs![0].data.shape).c(gy)];
  }
}

export class Add extends NNFunction {
  async forward([lhs, rhs]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.add(lhs as CPUTensor, rhs as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const gyShape = gy.data.shape;
    const lhsShape = this.inputs![0].data.shape;
    const rhsShape = this.inputs![1].data.shape;
    if (arrayEqual(lhsShape, rhsShape)) {
      // TODO: インスタンス共有してよいか確認
      return [gy, gy];
    } else {
      let glhs: Variable, grhs: Variable;
      if (arrayEqual(lhsShape, gyShape)) {
        glhs = gy;
      } else {
        glhs = await new SumTo(lhsShape).c(gy);
      }
      if (arrayEqual(rhsShape, gyShape)) {
        grhs = gy;
      } else {
        grhs = await new SumTo(rhsShape).c(gy);
      }
      return [glhs, grhs];
    }
  }
}

export class Mul extends NNFunction {
  async forward([lhs, rhs]: Tensor[]): Promise<Tensor[]> {
    return [CPUTensor.mul(lhs as CPUTensor, rhs as CPUTensor)];
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    if (!this.inputs) {
      throw new Error();
    }
    const [lhs, rhs] = this.inputs;
    const glhs = await new Mul().c(gy, rhs);
    const grhs = await new Mul().c(gy, lhs);
    return [glhs, grhs];
  }
}

export abstract class Layer {
  *parameters(recursive = true): Generator<Parameter> {
    for (const [, value] of Object.entries(this)) {
      if (value instanceof Parameter) {
        yield value;
      } else if (recursive && value instanceof Layer) {
        for (const subParam of value.parameters()) {
          yield subParam;
        }
      }
    }
  }

  *parametersWithName(
    recursive = true
  ): Generator<{ name: string; parameter: Parameter }> {
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Parameter) {
        yield { name, parameter: value };
      } else if (recursive && value instanceof Layer) {
        for (const subParam of value.parametersWithName()) {
          yield {
            name: `${name}.${subParam.name}`,
            parameter: subParam.parameter,
          };
        }
      }
    }
  }

  cleargrads(): void {
    for (const parameter of this.parameters()) {
      parameter.cleargrad();
    }
  }

  async call(...inputs: Variable[]): Promise<Variable[]> {
    const outputs = await this.forward(inputs);
    return outputs;
  }

  /**
   * Syntax suger for call() with only one output variable.
   * @param inputs Input variables.
   * @returns Promise of output variable.
   */
  async c(...inputs: Variable[]): Promise<Variable> {
    const outputs = await this.call(...inputs);
    if (outputs.length !== 1) {
      throw new Error(
        'c() can only be used for Layer with one output variable.'
      );
    }
    return outputs[0];
  }

  abstract forward(inputs: Variable[]): Promise<Variable[]>;
}

export abstract class Optimizer {
  params: Parameter[];

  constructor(params: Iterable<Parameter>) {
    this.params = Array.from(params);
  }

  zeroGrad() {
    for (const p of this.params) {
      p.cleargrad();
    }
  }

  abstract stepOne(parameter: Parameter): Promise<void>;

  async step() {
    for (const p of this.params) {
      await this.stepOne(p);
    }
  }
}

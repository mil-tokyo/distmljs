import { Backend } from '../backend';
import { defaultNNContext } from '../context';
import { CPUTensor } from '../tensor/cpu/cpuTensor';
import { Tensor } from '../tensor/tensor';
import { genCall } from '../tensor/tensorTypeUtil';
import { arrayEqual, nonNull } from '../util';

/**
 * Variable in neural network. Holds link for backpropagation.
 */
export class Variable {
  /**
   * Gradient of the variable. Computed by Variable.backward().
   */
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

  /**
   * Runs backpropagation.
   * @param retainGrad retains gradient of non-leaf variable.
   * @param createGraph create backpropagation graph in backpropagation (e.g. second-order derivative)
   */
  async backward(retainGrad = false, createGraph = false): Promise<void> {
    if (!this.grad) {
      const t = this.data.getClass().ones(this.data.shape, this.data.dtype);
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
      // TODO: nonNullチェック（batchnormのrunning_mean出力に勾配がないがbackwardを通すためチェックを外している）
      // differentialでない出力があるfunction全般における勾配の扱いを整理する必要あり
      const gys: Variable[] = f.outputs.map(
        (wv) => wv.deref()?.grad as Variable
      );

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

/**
 * Variable that is holded in Layer.
 */
export class Parameter extends Variable {
  constructor(
    public data: Tensor,
    public name?: string,
    public optimizable = true
  ) {
    super(data, name);
  }
}

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

  async forward(x: Tensor[]): Promise<Tensor[]> {
    return genCall(x, {
      all: (c, [t]) => [c.broadcastTo(t, this.shape)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new SumTo(nonNull(this.inputs)[0].data.shape).c(gy)];
  }
}

export class SumTo extends NNFunction {
  constructor(public shape: ReadonlyArray<number>) {
    super();
  }

  async forward(x: Tensor[]): Promise<Tensor[]> {
    return genCall(x, {
      all: (c, [t]) => [c.sumTo(t, this.shape)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new BroadcastTo(nonNull(this.inputs)[0].data.shape).c(gy)];
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
    return genCall([x], {
      all: (c, [t]) => [c.sum(t, this.axis, this.keepdims)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    return [await new BroadcastTo(nonNull(this.inputs)[0].data.shape).c(gy)];
  }
}

export class Add extends NNFunction {
  async forward([lhs, rhs]: Tensor[]): Promise<Tensor[]> {
    return genCall([lhs, rhs], {
      all: (c, [lhs, rhs]) => [c.add(lhs, rhs)],
    });
  }

  async backward([gy]: Variable[]): Promise<Variable[]> {
    const gyShape = gy.data.shape;
    const lhsShape = nonNull(this.inputs)[0].data.shape;
    const rhsShape = nonNull(this.inputs)[1].data.shape;
    if (arrayEqual(lhsShape, rhsShape)) {
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
    return genCall([lhs, rhs], {
      all: (c, [lhs, rhs]) => [c.mul(lhs, rhs)],
    });
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
  training = true; // default value is same as PyTorch

  *parameters(recursive = true, optimizableOnly = true): Generator<Parameter> {
    for (const [, value] of Object.entries(this)) {
      if (value instanceof Parameter) {
        if (!optimizableOnly || value.optimizable) {
          yield value;
        }
      } else if (recursive && value instanceof Layer) {
        for (const subParam of value.parameters(recursive, optimizableOnly)) {
          yield subParam;
        }
      }
    }
  }

  *parametersWithName(
    recursive = true,
    optimizableOnly = true
  ): Generator<{ name: string; parameter: Parameter }> {
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Parameter) {
        if (!optimizableOnly || value.optimizable) {
          yield { name, parameter: value };
        }
      } else if (recursive && value instanceof Layer) {
        for (const subParam of value.parametersWithName(
          recursive,
          optimizableOnly
        )) {
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

  /**
   * Moves backend of tensor inside the Layer. This is in-place operation.
   * @param backend backend to move to.
   */
  async to(backend: Backend): Promise<void> {
    for (const param of this.parameters(true, false)) {
      param.data = await param.data.to(backend);
    }
  }

  /**
   * Sets the model's training mode
   * @param mode true (default): training mode, false: evaluation mode
   */
  train(mode = true): void {
    this.training = mode;
    // apply recursively
    for (const value of Object.values(this)) {
      if (value instanceof Layer) {
        value.train(mode);
      }
    }
  }

  /**
   * Sets the model mode to eval
   */
  eval(): void {
    this.train(false);
  }

  async stateMap(): Promise<Map<string, CPUTensor>> {
    const m = new Map<string, CPUTensor>();
    for (const { name, parameter } of this.parametersWithName(true, false)) {
      m.set(name, await parameter.data.to('cpu'));
    }
    return m;
  }

  async setStateMap(
    map: Map<string, CPUTensor>,
    strict = true
  ): Promise<{ missingKeys: string[]; unexpectedKeys: string[] }> {
    const unexpectedKeys = new Set(map.keys());
    const missingKeys: string[] = [];
    for (const { name, parameter } of this.parametersWithName(true, false)) {
      const v = map.get(name);
      unexpectedKeys.delete(name);
      if (v) {
        const backend = parameter.data.backend;
        parameter.data.dispose();
        parameter.data = await v.to(backend);
      } else {
        if (strict) {
          throw new Error(`Tensor for ${name} is missing.`);
        } else {
          missingKeys.push(name);
        }
      }
    }
    if (strict) {
      if (unexpectedKeys.size > 0) {
        throw new Error(`Tensor ${Array.from(unexpectedKeys)} are unexpected.`);
      }
    }
    return { unexpectedKeys: Array.from(unexpectedKeys), missingKeys };
  }
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

  abstract getKeepTensors(): Tensor[];
}

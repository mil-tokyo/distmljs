import * as K from 'kakiage';
import Variable = K.nn.Variable;

export class NetMLP extends K.nn.core.Layer {
    fc1: K.nn.layers.Linear;
    fc2: K.nn.layers.Linear;
    fc3: K.nn.layers.Linear;

    constructor(inputShape: number, nClasses: number, nch: number = 256) {
        super();
        
        this.fc1 = new K.nn.layers.Linear(inputShape, nch);
        this.fc2 = new K.nn.layers.Linear(nch, nch);
        this.fc3 = new K.nn.layers.Linear(nch, nClasses);
    }

    async forward(inputs: Variable[]): Promise<Variable[]> {
        let y = inputs[0];
        y = await this.fc1.c(y);
        y = await K.nn.functions.relu(y);
        y = await this.fc2.c(y);
        y = await K.nn.functions.relu(y);
        y = await this.fc3.c(y);
        return [y];
    }
}

export class DoubleClippedNetMLP extends K.nn.core.Layer {
    q1: NetMLP;
    q2: NetMLP;

    constructor(inputShape: number, nClasses: number, nch: number = 256) {
        super();
        this.q1 = new NetMLP(inputShape, nClasses, nch);
        this.q2 = new NetMLP(inputShape, nClasses, nch);
    }

    async forward(inputs: Variable[]): Promise<Variable[]> {
        let x = inputs[0]
        let y1, y2;
        y1 = await this.q1.c(x);
        y2 = await this.q2.c(x);
        return [y1, y2];
    }
}


export function makeModel(
    name: string,
    inputShape: number,
    nClasses: number
  ): K.nn.core.Layer {
    switch (name) {
      case 'mlp':
        return new NetMLP(inputShape, nClasses);
      case 'dmlp':
        return new DoubleClippedNetMLP(inputShape, nClasses);
      default:
        throw new Error(`Unknown model ${name}`);
    }
  }
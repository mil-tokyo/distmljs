import Chart from 'chart.js/auto';
import * as K from 'kakiage';
import CPUTensor = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import VariableResolvable = K.nn.VariableResolvable;

function print(message: string, time = false): void {
  const div = document.getElementById('result');
  const elem = document.createElement('div');
  let m = message;
  if (time) {
    m = `${performance.now() | 0}ms: ${m}`;
  }
  elem.innerText = m;
  div?.appendChild(elem);
}

function wait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 10);
  });
}

/**
 * Make synthesized dataset of sin curve.
 * @param size
 * @param random
 * @returns
 */
function makeDataset(size: number): [CPUTensor, CPUTensor] {
  // 0, 0.001, 0.002, ..., 1.0
  const x = CPUTensor.fromArray(
    new Array(size).fill(0).map((_, i) => i / size),
    [size, 1]
  );
  const y = CPUTensor.sin(CPUTensor.mul(x, CPUTensor.s(10)));
  return [x, y];
}

class MLPModel extends K.nn.core.Layer {
  l1: K.nn.layers.Linear;
  l2: K.nn.layers.Linear;

  constructor(inFeatures: number, hidden: number, outFeatures: number) {
    super();
    this.l1 = new K.nn.layers.Linear(inFeatures, hidden);
    this.l2 = new K.nn.layers.Linear(hidden, outFeatures);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y: VariableResolvable = inputs[0];
    y = this.l1.c(y);
    y = K.nn.functions.relu(y);
    y = this.l2.c(y);
    return [await y];
  }
}

async function scalarTrain() {
  const size = 1000;
  const hidden = 64;
  const lr = 0.01;
  const [trainX, trainY] = makeDataset(size);
  const model = new MLPModel(1, hidden, 1);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr, 0.9);
  // Training by gradient descent (batch = whole training data)
  for (let i = 0; i < 2000; i++) {
    // tidy: release memory allocated inside the block when the block ends.
    await K.tidy(async () => {
      const y = model.c(new K.nn.Variable(trainX));
      const loss = await K.nn.functions.mseLoss(y, new K.nn.Variable(trainY));

      if (i % 100 === 0) {
        const lossScalar = (loss.data as CPUTensor).get(0);
        print(`iteration: ${i}, loss: ${lossScalar}`);
        await wait();
      }
      optimizer.zeroGrad();
      await loss.backward();
      await optimizer.step();

      // keep tensors inside model and optimizer after this block ends.
      return [model, optimizer];
    });
  }
  const testPredict = await model.c(new K.nn.Variable(trainX));

  const ctx = (
    document.getElementById('myChart')! as HTMLCanvasElement
  ).getContext('2d')!;
  const _ = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trainX.toArray(),
      datasets: [
        {
          label: 'Predicted',
          data: (testPredict.data as CPUTensor).toArray(),
          pointBackgroundColor: '#f00',
          pointBorderWidth: 0,
          backgroundColor: '#f00',
          borderWidth: 1,
        },
        {
          label: 'Ground truth',
          data: trainY.toArray(),
          pointBackgroundColor: '#00f',
          pointBorderWidth: 0,
          backgroundColor: '#00f',
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: { suggestedMax: 1.5, suggestedMin: -1.5 },
        x: { display: false },
      },
    },
  });
}

window.addEventListener('load', () => {
  document.getElementById('run')!.onclick = () => scalarTrain();
});

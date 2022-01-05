import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import Random = K.math.Random;

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

function hello() {
  const x = T.fromArray([1, 3, 5]);
  const y = T.fromArray([20, 40, 60]);
  const z = T.add(x, y);
  print(`${x.toArray()} + ${y.toArray()} = ${z.toArray()}`, true);
}

function makeDataset(size: number, random: Random) {
  const x = T.fromArray(random.random(size), [size, 1]);
  const noise = T.mul(T.fromArray(random.normal(size), [size, 1]), T.s(0.1));
  const y = T.add(T.sin(T.mul(x, T.s(10))), noise);
  console.log(x.toArray());
  console.log(y.toArray());
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
    let y = inputs[0];
    y = await this.l1.c(y);
    y = await K.nn.functions.relu(y);
    y = await this.l2.c(y);
    return [y];
  }
}

async function scalarTrain() {
  const random = new Random(0);
  const size = 1000;
  const hidden = 32;
  const lr = 0.01;
  const [trainX, trainY] = makeDataset(size, random);
  const [testX, testY] = makeDataset(size, random);
  const model = new MLPModel(1, hidden, 1);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr);
  for (let i = 0; i < 10000; i++) {
    const y = await model.c(new K.nn.Variable(trainX));
    const loss = await K.nn.functions.mseLoss(y, new K.nn.Variable(trainY));
    const lossScalar = (loss.data as T).get(0);
    if (i % 100 === 0) {
      print(`loss: ${lossScalar}`);
    }
    optimizer.zeroGrad();
    await loss.backward();
    await optimizer.step();
  }
  const testPredict = await model.c(new K.nn.Variable(testX));
  console.log((testX as T).toArray());
  console.log((testY as T).toArray());
  console.log((testPredict.data as T).toArray());
  const testLoss = await K.nn.functions.mseLoss(
    testPredict,
    new K.nn.Variable(testY)
  );
  const testLossScalar = (testLoss.data as T).get(0);
  print(`test loss: ${testLossScalar}`, true);
}

window.addEventListener('load', () => {
  hello();
  scalarTrain();
});

import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import FetchDataset = K.dataset.datasets.FetchDataset;
import DataLoader = K.dataset.DataLoader;

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

function wait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 10);
  });
}

async function train() {
  const trainDataset = new FetchDataset(
    './dataset/mnist_preprocessed_flatten_train.bin'
  );
  await trainDataset.load();
  const testDataset = new FetchDataset(
    './dataset/mnist_preprocessed_flatten_test.bin'
  );
  await testDataset.load();
  const trainLoader = new DataLoader(trainDataset, { batchSize: 32 });
  const testLoader = new DataLoader(testDataset, { batchSize: 32 });

  const hidden = 32;
  const lr = 0.01;
  const model = new MLPModel(784, hidden, 10);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr);

  for (let epoch = 0; epoch < 3; epoch++) {
    print(`epoch ${epoch}`);
    let trainIter = 0;
    for await (const [images, labels] of trainLoader) {
      const y = await model.c(new K.nn.Variable(images));
      const loss = await K.nn.functions.softmaxCrossEntropy(
        y,
        new K.nn.Variable(labels)
      );
      const lossScalar = (loss.data as T).get(0);
      if (trainIter % 100 === 0) {
        print(`loss: ${lossScalar}`, true);
        await wait();
      }
      optimizer.zeroGrad();
      await loss.backward();
      await optimizer.step();
      trainIter++;
    }

    let nTestSamples = 0;
    let nCorrect = 0;
    for await (const [images, labels] of testLoader) {
      // TODO: no grad
      const y = await model.c(new K.nn.Variable(images));
      // TOOD: accuracy utility
      const pred = y.data as T;
      for (let i = 0; i < pred.shape[0]; i++) {
        let maxLogit = -Infinity;
        let maxLabel = 0;
        for (let j = 0; j < pred.shape[1]; j++) {
          const logit = pred.get(i, j);
          if (logit > maxLogit) {
            maxLogit = logit;
            maxLabel = j;
          }
        }

        nTestSamples++;
        if (maxLabel === labels.get(i)) {
          nCorrect++;
        }
      }
    }
    const accuracy = nCorrect / nTestSamples;
    print(`Test accuracy: ${accuracy}, ${nTestSamples}, ${nCorrect}`, true);
  }
  print('train end');
}

window.addEventListener('load', () => {
  document.getElementById('start-training')!.onclick = async () => {
    await train();
  };
});

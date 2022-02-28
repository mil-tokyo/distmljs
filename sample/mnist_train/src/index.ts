import * as K from 'kakiage';
import Variable = K.nn.Variable;
import FetchDataset = K.dataset.datasets.FetchDataset;
import DataLoader = K.dataset.DataLoader;

const localStorageKey = 'localstorage://kakiage/mnistTrain';

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

let model: MLPModel;

async function train(backend: K.Backend) {
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

  const lr = 0.01;
  await model.to(backend);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr);

  print(`Start training on backend ${backend}`);
  for (let epoch = 0; epoch < 1; epoch++) {
    print(`epoch ${epoch}`);
    let trainIter = 0;
    for await (const [images, labels] of trainLoader) {
      await K.tidy(async () => {
        const y = await model.c(new K.nn.Variable(await images.to(backend)));
        const loss = await K.nn.functions.softmaxCrossEntropy(
          y,
          new K.nn.Variable(await labels.to(backend))
        );
        if (trainIter % 100 === 0) {
          const lossScalar = (await loss.data.to('cpu')).get(0);
          print(`loss: ${lossScalar}`, true);
          await wait();
        }
        optimizer.zeroGrad();
        await loss.backward();
        await optimizer.step();
        trainIter++;
        return [model, optimizer];
      });
    }

    let nTestSamples = 0;
    let nCorrect = 0;
    for await (const [images, labels] of testLoader) {
      await K.tidy(async () => {
        // TODO: no grad
        const y = await model.c(new K.nn.Variable(await images.to(backend)));
        // TOOD: accuracy utility
        const pred = await y.data.to('cpu');
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
        return [model];
      });
    }
    const accuracy = nCorrect / nTestSamples;
    print(`Test accuracy: ${accuracy}, ${nTestSamples}, ${nCorrect}`, true);
  }
  print('train end');
}

async function startTraining(load?: 'localStorage' | 'file') {
  try {
    const backend =
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      (
        document.querySelector(
          'input[name="backend"]:checked'
        )! as HTMLInputElement
      ).value as K.Backend;
    try {
      if (backend === 'webgl') {
        await K.tensor.initializeNNWebGLContext();
      }
      if (backend === 'webgpu') {
        await K.tensor.initializeNNWebGPUContext();
      }
    } catch (error) {
      alert(`Failed to initialize backend ${backend}. ${error}`);
      return;
    }

    const hidden = 32;
    model = new MLPModel(784, hidden, 10);
    const des = new K.tensor.TensorDeserializer();
    if (load === 'localStorage') {
      await model.setStateMap(await des.fromLocalStorage(localStorageKey));
    } else if (load === 'file') {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      const files = (
        document.getElementById('upload-file')! as HTMLInputElement
      ).files;
      if (!files || !files[0]) {
        alert('No file specified');
        return;
      }
      const file = files[0];
      await model.setStateMap(await des.fromFile(file));
    }
    await train(backend);
  } catch (error) {
    console.error(error);
    alert(`Training failed ${(error as Error).message}`);
  }
}

window.addEventListener('load', () => {
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  document.getElementById('start-training')!.onclick = () => startTraining();
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  document.getElementById('start-training-with-saved')!.onclick = () =>
    startTraining('localStorage');
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  document.getElementById('start-training-with-file')!.onclick = () =>
    startTraining('file');
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  document.getElementById('save-inside-browser')!.onclick = async () => {
    try {
      const ser = new K.tensor.TensorSerializer();
      await ser.toLocalStorage(await model.stateMap(), localStorageKey);
    } catch (error) {
      console.error(error);
      alert(`Save failed ${(error as Error).message}`);
    }
  };
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  document.getElementById('download-to-file')!.onclick = async () => {
    try {
      const ser = new K.tensor.TensorSerializer();
      await ser.toFile(await model.stateMap(), 'mnisttrain.bin');
    } catch (error) {
      console.error(error);
      alert(`Save failed ${(error as Error).message}`);
    }
  };
});

import Chart from 'chart.js/auto';
import * as K from 'kakiage';
import Variable = K.nn.Variable;
import VariableResolvable = K.nn.VariableResolvable;
import CPUTensor = K.tensor.CPUTensor;
import FetchDataset = K.dataset.datasets.FetchDataset;
import DataLoader = K.dataset.DataLoader;

const localStorageKey = 'localstorage://kakiage/mnistTrain';

function status(message: string): void {
  document.getElementById('status')!.innerText = message;
}

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

// define model by extending K.nn.core.Layer
class MLPModel extends K.nn.core.Layer {
  // trainable layers have to be stored as property
  l1: K.nn.layers.Linear;
  l2: K.nn.layers.Linear;

  constructor(inFeatures: number, hidden: number, outFeatures: number) {
    super();
    this.l1 = new K.nn.layers.Linear(inFeatures, hidden);
    this.l2 = new K.nn.layers.Linear(hidden, outFeatures);
  }

  // forward defines control flow
  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y: VariableResolvable = inputs[0];
    y = this.l1.c(y);
    y = K.nn.functions.relu(y);
    y = this.l2.c(y);
    return [await y];
  }
}

function wait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 10);
  });
}

let model: MLPModel;

async function train(backend: K.Backend) {
  status(`Loading dataset`);
  // load dataset file created by prepare_dataset.py
  const trainDataset = new FetchDataset(
    './dataset/mnist_preprocessed_flatten_train.bin'
  );
  await trainDataset.load();
  const testDataset = new FetchDataset(
    './dataset/mnist_preprocessed_flatten_test.bin'
  );
  await testDataset.load();
  // create dataset iterator
  const trainLoader = new DataLoader(trainDataset, { batchSize: 32 });
  const testLoader = new DataLoader(testDataset, { batchSize: 32 });

  const lr = 0.01;
  await model.to(backend);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr);

  // for loss curve visualization
  const ctx = (
    document.getElementById('myChart')! as HTMLCanvasElement
  ).getContext('2d')!;
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Train loss',
          showLine: true,
          pointStyle: 'rect',
          data: [] as { x: number; y: number }[],
          pointBackgroundColor: '#f00',
          pointBorderWidth: 0,
          backgroundColor: '#f00',
          borderColor: '#f00',
          borderWidth: 1,
        },
        {
          label: 'Test loss',
          showLine: true,
          data: [] as { x: number; y: number }[],
          pointStyle: 'rect',
          pointBackgroundColor: '#00f',
          pointBorderWidth: 0,
          backgroundColor: '#00f',
          borderColor: '#00f',
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: {
          suggestedMax: 1.0,
          suggestedMin: 0,
          title: { display: true, text: 'loss' },
        },
        x: {
          title: { display: true, text: 'training steps' },
        },
      },
    },
  });

  print(`Start training on backend ${backend}`);
  let totalTrainIter = 0;
  for (let epoch = 0; epoch < 2; epoch++) {
    const epochStartTime = Date.now();
    print(`epoch ${epoch}`);
    let trainIter = 0;
    // training loop.
    for await (const [images, labels] of trainLoader) {
      // K.tidy: takes function, and releases tensor allocated in the function after it ends.
      // it is needed because GPU memory cannot be garbage collected.
      await K.tidy(async () => {
        // model (K.nn.core.Layer) can be called with model.c
        // model receives Variable, which can be constructed with new Variable(Tensor)
        // Tensor.to(backend) copies tensor data to another backend (CPUTensor/WebGLTensor/WebGPUTensor)
        const y = await model.c(new K.nn.Variable(await images.to(backend)));
        const loss = await K.nn.functions.softmaxCrossEntropy(
          y,
          new K.nn.Variable(await labels.to(backend))
        );
        if (trainIter % 100 === 0) {
          const now = Date.now();
          const speed = (now - epochStartTime) / (trainIter + 1);
          // note: this is not precise because it includes wait
          status(`Training iter: ${trainIter}, ${speed.toFixed(2)} ms / iter`);
          // only CPUTensor has get method.
          const lossScalar = (await loss.data.to('cpu')).get(0);
          // plot loss
          chart.data.datasets[0].data.push({
            x: totalTrainIter,
            y: lossScalar,
          });
          chart.update('none');
          // insert wait to process UI event.
          await wait();
        }

        // remove gradient of Parameter in model.
        optimizer.zeroGrad();
        // backpropagation
        await loss.backward();
        // update Parameter in model using gradient.
        await optimizer.step();
        trainIter++;
        totalTrainIter++;
        // Tensors inside them are kept after tidy.
        return [model, optimizer];
      });
    }

    let nTestSamples = 0;
    let nCorrect = 0;
    let sumLoss = 0;
    status(`Running validation`);
    // don't keep tensor needed for backpropagation
    await K.context.defaultNNContext.withValue(
      'enableBackprop',
      false,
      async () => {
        for await (const [images, labels] of testLoader) {
          await K.tidy(async () => {
            const y = await model.c(
              new K.nn.Variable(await images.to(backend))
            );
            const loss = await K.nn.functions.softmaxCrossEntropy(
              y,
              new K.nn.Variable(await labels.to(backend))
            );
            const lossScalar = (await loss.data.to('cpu')).get(0);
            sumLoss += lossScalar * labels.shape[0];
            const pred = CPUTensor.argmax(await y.data.to('cpu'), 1); // predicted label
            for (let i = 0; i < pred.shape[0]; i++) {
              nTestSamples++;
              if (pred.get(i) === labels.get(i)) {
                nCorrect++;
              }
            }
            return [model];
          });
        }
      }
    );
    const accuracy = nCorrect / nTestSamples;
    const avgLoss = sumLoss / nTestSamples;
    chart.data.datasets[1].data.push({
      x: totalTrainIter,
      y: avgLoss,
    });
    chart.update('none');
    print(`Test accuracy: ${accuracy}, loss: ${avgLoss}`);
  }
  print('Training finished');
  document.getElementById('save-control')!.style.display = 'block';
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
      // model.stateMap() returns model parameters with its keys
      // toLocalStorage saves to localStorage (inside browser)
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
      // download file as if right-click on link and click save link as
      await ser.toFile(await model.stateMap(), 'mnisttrain.bin');
    } catch (error) {
      console.error(error);
      alert(`Save failed ${(error as Error).message}`);
    }
  };
});

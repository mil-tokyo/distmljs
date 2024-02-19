import Chart from 'chart.js/auto';
import * as K from 'kakiage';
import { throttle } from 'lodash';
import { ResNet18 } from './model';
import CPUTensor = K.tensor.CPUTensor;
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

const status = throttle((message: string) => {
  document.getElementById('status')!.innerText = message;
}, 1000);

function wait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 10);
  });
}

async function train(backend: K.Backend, batchSize: number) {
  status(`Loading dataset`);
  const trainDataset = new FetchDataset(
    './dataset/cifar10_preprocessed_train.bin'
  );
  await trainDataset.load();
  const testDataset = new FetchDataset(
    './dataset/cifar10_preprocessed_test.bin'
  );
  await testDataset.load();
  const trainLoader = new DataLoader(trainDataset, { batchSize });
  const testLoader = new DataLoader(testDataset, { batchSize });

  const lr = 0.01 * (batchSize / 32); // lower learning rate when batch size is small
  const model = new ResNet18(10);
  await model.to(backend);
  const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr);

  let lastWaited = Date.now();
  const waitIfNeeded = async () => {
    const now = Date.now();
    if (now - lastWaited >= 1000) {
      await wait();
      lastWaited = now;
    }
  };

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
  const totalEpochs = 3;
  for (let epoch = 1; epoch <= totalEpochs; epoch++) {
    const epochStartTime = Date.now();
    print(`epoch ${epoch} / ${totalEpochs}`);
    const trainStartTime = Date.now();
    let trainIter = 0;
    model.train();
    for await (const [images, labels] of trainLoader) {
      await K.tidy(async () => {
        const y = await model.c(new K.nn.Variable(await images.to(backend)));
        const loss = await K.nn.functions.softmaxCrossEntropy(
          y,
          new K.nn.Variable(await labels.to(backend))
        );

        const lossScalar = (await loss.data.to('cpu')).get(0);
        const now = Date.now();
        const speed = (now - epochStartTime) / (trainIter + 1);
        // note: this is not precise because it includes wait
        status(
          `Training iter: ${trainIter} / ${trainLoader.length}, ${speed.toFixed(
            2
          )} ms / iter, loss: ${lossScalar}, learning rate: ${lr}`
        );

        if (trainIter % 100 === 0) {
          // plot loss
          chart.data.datasets[0].data.push({
            x: totalTrainIter,
            y: lossScalar,
          });
          chart.update('none');
        }
        optimizer.zeroGrad();
        await loss.backward();
        await optimizer.step();
        trainIter++;
        totalTrainIter++;
        return [model, optimizer];
      });
      await waitIfNeeded();
    }

    const trainEndTime = Date.now();
    print(
      `epoch ${epoch} finished in ${(trainEndTime - trainStartTime) / 1000}s (${trainIter} iterations, ${trainIter * batchSize} samples)`
    );

    let nTestSamples = 0;
    let nCorrect = 0;
    let sumLoss = 0;
    model.eval();
    status(`Running validation`);
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
          await waitIfNeeded();
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
}

window.addEventListener('load', () => {
  document.getElementById('start-training')!.onclick = async () => {
    const backend = (
      document.querySelector(
        'input[name="backend"]:checked'
      )! as HTMLInputElement
    ).value as K.Backend;
    const batchSize = Number(
      (document.querySelector('input[name="batch-size"]')! as HTMLInputElement)
        .value
    );
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
    await train(backend, batchSize);
  };
});

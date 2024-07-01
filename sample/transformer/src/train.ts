import * as K from 'distmljs';
import CPUTensor = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;
import { TransformerModel } from './transformer';
import { generateSquareSubsquentMask, log, wait } from './common';

const bptt = 10;

function getBatch(source: CPUTensor, i: number): [CPUTensor, CPUTensor] {
  const seqLen = Math.min(bptt, source.shape[0] - 1 - i);
  const data = source.gets(K.slice(i, i + seqLen));
  const target = source.gets(K.slice(i + 1, i + 1 + seqLen)).reshape([-1]);
  return [data, target];
}

let model: TransformerModel;

export async function startTraining() {
  try {
    const trainDataset = await new TensorDeserializer().fromHTTP(
      './dataset/wikitext2_train.bin'
    );
    const trainData = trainDataset.get('data')!;

    model = new TransformerModel(28782, 64, 2, 64, 2, 0.2);
    const lr = 5.0;
    const optimizer = new K.nn.optimizers.SGD(model.parameters(), lr, 0.0);
    const srcMask = generateSquareSubsquentMask(bptt);
    const startTime = Date.now();
    for (let epoch = 0; epoch < 1; epoch++) {
      model.train();
      optimizer.lr = lr * 0.95 ** epoch; // LR step
      const trainBatches = Math.floor((trainData.shape[0] - 1) / bptt);
      for (let i = 0; i < trainBatches; i++) {
        await K.tidy(async () => {
          const [data, targets] = getBatch(trainData, i);
          optimizer.zeroGrad();
          const output = await model.c(
            new Variable(data),
            new Variable(srcMask)
          );
          const loss = await K.nn.functions.softmaxCrossEntropy(
            await K.nn.functions.reshape(output, [
              -1,
              output.data.shape[output.data.shape.length - 1],
            ]),
            new Variable(targets)
          );
          const [lossValue] = await loss.data.toArrayAsync();
          log(
            `iter ${i}/${trainBatches} loss ${lossValue} ${Date.now() - startTime
            } ms`
          );
          await loss.backward();
          K.nn.utils.clipGradNorm_(model.parameters(), 0.5);
          await optimizer.step();
          return [model, optimizer];
        });
        await wait();
      }
    }
  } catch (error) {
    console.error(error);
    alert((error as Error).message);
  }
}

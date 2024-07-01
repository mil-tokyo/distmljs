// gradient checking
import * as K from 'distmljs';
import CPUTensor = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;
import { TransformerModel } from './transformer';
import { log } from './common';

let model: TransformerModel;

function compareTensor(name: string, actual: CPUTensor, expected: CPUTensor) {
  if (actual.ndim !== expected.ndim) {
    log(`${name}: shape mismatch ${actual.shape} !== ${expected.shape}`);
    return;
  }
  for (let i = 0; i < actual.shape.length; i++) {
    if (actual.shape[i] !== expected.shape[i]) {
      log(`${name}: shape mismatch ${actual.shape} !== ${expected.shape}`);
      return;
    }
  }
  const aa = actual.buffer.data;
  const ea = expected.buffer.data;
  let maxDiff = 0;
  let diffPos = 0;
  for (let i = 0; i < aa.length; i++) {
    const a = aa[i];
    const e = ea[i];
    const diff = Math.abs(a - e);
    if (diff > maxDiff) {
      maxDiff = diff;
      diffPos = i;
    }
  }
  if (maxDiff > 1e-3) {
    log(
      `${name}: diff=${maxDiff} at ${diffPos}, actual=${aa[diffPos]}, expected=${ea[diffPos]}`
    );
  } else {
    console.info(
      `${name}: diff=${maxDiff} at ${diffPos}, actual=${aa[diffPos]}, expected=${ea[diffPos]}`
    );
  }
}

export async function checkGrad() {
  try {
    model = new TransformerModel(28782, 64, 2, 64, 2, 0.2);
    const nameShape: { name: string; shape: readonly number[] }[] = [];
    for (const { name, parameter } of model.parametersWithName(true, false)) {
      nameShape.push({ name, shape: parameter.data.shape });
    }
    nameShape.sort((a, b) => a.name.localeCompare(b.name));
    log('loading weight');
    const trainedWeights = await new TensorDeserializer().fromHTTP(
      './dataset/pytorch_trained_weight.bin'
    );
    await model.setStateMap(trainedWeights);
    const testSample = await new TensorDeserializer().fromHTTP(
      './dataset/pytorch_sample_batch.bin'
    );
    const ptGrads = await new TensorDeserializer().fromHTTP(
      './dataset/pytorch_trained_grad.bin'
    );
    log('evaluating');
    await K.tidy(async () => {
      model.eval();
      const output = await model.c(
        new Variable(testSample.get('data')!),
        new Variable(testSample.get('src_mask')!)
      );
      const ce = await K.nn.functions.softmaxCrossEntropy(
        await K.nn.functions.reshape(output, [
          -1,
          output.data.shape[output.data.shape.length - 1],
        ]),
        new Variable(testSample.get('targets')!)
      );
      return [];
    });
    await K.tidy(async () => {
      model.eval(); // disable dropout for deterministic result
      const output = await model.c(
        new Variable(testSample.get('data')!),
        new Variable(testSample.get('src_mask')!)
      );
      const ce = await K.nn.functions.softmaxCrossEntropy(
        await K.nn.functions.reshape(output, [
          -1,
          output.data.shape[output.data.shape.length - 1],
        ]),
        new Variable(testSample.get('targets')!)
      );
      await ce.backward();
      for (const { name, parameter } of model.parametersWithName(true, true)) {
        compareTensor(
          name,
          ptGrads.get(name)!,
          await parameter.grad!.data.to('cpu')
        );
      }
      return [];
    });
    log('Check complete');
  } catch (error) {
    console.error(error);
    alert(`Check failed ${(error as Error).message}`);
  }
}

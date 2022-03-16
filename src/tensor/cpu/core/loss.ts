import { arrayEqual, arrayProd } from '../../../util';
import { CPUTensor } from '../cpuTensor';

export function nllLoss(x: CPUTensor, label: CPUTensor): CPUTensor {
  const [batch, cs] = x.shape;
  if (x.shape.length !== 2) {
    throw new Error('nllLoss needs 2d input');
  }
  if (label.shape.length !== 1) {
    throw new Error('nllLoss needs 1d label input');
  }
  const output = CPUTensor.zeros([]);
  const dx = x.getBuffer().data;
  const dlabel = label.getBuffer().data;
  let ceSum = 0.0;
  for (let b = 0; b < batch; b++) {
    const label = dlabel[b];
    ceSum += -Math.log(dx[b * cs + label]);
  }
  const ceAvg = ceSum / batch;
  output.getBuffer().data[0] = ceAvg;
  return output;
}

export function softmax(x: CPUTensor): CPUTensor {
  if (x.shape.length < 2) {
    throw new Error('softmax needs 2d input');
  }
  const batch = arrayProd(x.shape.slice(0, x.shape.length - 1));
  const cs = x.shape[x.shape.length - 1];
  const output = CPUTensor.zeros(x.shape);
  const dx = x.getBuffer().data;
  const dy = output.getBuffer().data;
  for (let b = 0; b < batch; b++) {
    let max = -Infinity;
    for (let c = 0; c < cs; c++) {
      const v = dx[b * cs + c];
      if (v > max) {
        max = v;
      }
    }
    let expSum = 0.0;
    for (let c = 0; c < cs; c++) {
      const v = dx[b * cs + c] - max;
      const exp = Math.exp(v);
      dy[b * cs + c] = exp;
      expSum += exp;
    }
    for (let c = 0; c < cs; c++) {
      dy[b * cs + c] /= expSum;
    }
  }
  return output;
}

export function softmaxBackward(y: CPUTensor, gy: CPUTensor): CPUTensor {
  if (y.shape.length < 2) {
    throw new Error('softmaxBackward needs 2d input');
  }
  const batch = arrayProd(y.shape.slice(0, y.shape.length - 1));
  const cs = y.shape[y.shape.length - 1];
  const output = CPUTensor.zeros(y.shape);
  const dy = y.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const dgx = output.getBuffer().data;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < cs; c++) {
      let sum = 0.0;
      const my = dy[b * cs + c];
      for (let d = 0; d < cs; d++) {
        if (d === c) {
          sum += my * (1 - my) * dgy[b * cs + d];
        } else {
          sum += -my * dy[b * cs + d] * dgy[b * cs + d];
        }
      }
      dgx[b * cs + c] = sum;
    }
  }
  return output;
}

export function softmaxCrossEntropyBackward(
  softmax: CPUTensor,
  label: CPUTensor,
  gy: CPUTensor
): CPUTensor {
  // x -> softmax -> lossでgxを求める
  // TODO: labelはint32
  const [batch, cs] = softmax.shape;
  if (softmax.shape.length !== 2) {
    throw new Error('nllLoss needs 2d input');
  }
  if (label.shape.length !== 1) {
    throw new Error('nllLoss needs 1d label input');
  }
  if (!arrayEqual(gy.shape, [])) {
    throw new Error('gy must be scalar');
  }
  const output = CPUTensor.zeros(softmax.shape);
  const dgx = output.getBuffer().data;
  const dsoftmax = softmax.getBuffer().data;
  const dlabel = label.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const gyValue = dgy[0] / batch;
  for (let b = 0; b < batch; b++) {
    const label = dlabel[b];
    for (let c = 0; c < cs; c++) {
      let v = dsoftmax[b * cs + c];
      if (c === label) {
        v -= 1;
      }
      dgx[b * cs + c] = v * gyValue;
    }
  }
  return output;
}

export function reluBackprop(x: CPUTensor, gy: CPUTensor): CPUTensor {
  const output = CPUTensor.zeros(x.shape);
  const dx = x.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const dgx = output.getBuffer().data;
  for (let i = 0; i < output.size; i++) {
    dgx[i] = dx[i] > 0.0 ? dgy[i] : 0.0;
  }
  return output;
}

export function sigmoidBackprop(y: CPUTensor, gy: CPUTensor): CPUTensor {
  const output = CPUTensor.zeros(gy.shape);
  const dy = y.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const dgx = output.getBuffer().data;
  for (let i = 0; i < output.size; i++) {
    const yv = dy[i];
    dgx[i] = (1 - yv) * yv * dgy[i];
  }
  return output;
}

export function mseLoss(a: CPUTensor, b: CPUTensor): CPUTensor {
  if (!arrayEqual(a.shape, b.shape)) {
    throw new Error('Shape mismatch');
  }
  const output = CPUTensor.zeros([]);
  const da = a.getBuffer().data;
  const db = b.getBuffer().data;
  const dy = output.getBuffer().data;
  let sum = 0.0;
  for (let i = 0; i < a.size; i++) {
    const diff = da[i] - db[i];
    sum += diff * diff;
  }
  dy[0] = sum / a.size;
  return output;
}

export function mseLossBackprop(
  a: CPUTensor,
  b: CPUTensor,
  gy: CPUTensor
): [CPUTensor, CPUTensor] {
  if (!arrayEqual(a.shape, b.shape)) {
    throw new Error('Shape mismatch');
  }
  if (gy.ndim !== 0) {
    throw new Error('gy must be scalar');
  }
  const da = a.getBuffer().data;
  const db = b.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const ga = CPUTensor.zeros(a.shape);
  const gb = CPUTensor.zeros(a.shape);
  const dga = ga.getBuffer().data;
  const dgb = gb.getBuffer().data;
  const coef = (dgy[0] * 2) / a.size;
  for (let i = 0; i < a.size; i++) {
    const v = (da[i] - db[i]) * coef;
    dga[i] = v;
    dgb[i] = -v;
  }
  return [ga, gb];
}

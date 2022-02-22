import { slice } from '../..';
import { arange, arrayProd } from '../../../util';
import { sum } from '../core';
import { CPUTensor } from '../cpuTensor';

export interface BatchNormParams {
  axis: number;
  training: boolean;
  eps: number;
  momentum?: number;
  trackRunningStats: boolean;
}

export function batch_norm_cpu(
  x: CPUTensor,
  affine: { weight: CPUTensor; bias: CPUTensor } | null,
  runningStats: {
    runningMean: CPUTensor;
    runningVar: CPUTensor;
    numBatchesTracked: CPUTensor;
  } | null,
  params: BatchNormParams
): {
  y: CPUTensor;
  statsForBackprop: CPUTensor;
  updatedRunningStats: {
    runningMean: CPUTensor;
    runningVar: CPUTensor;
    numBatchesTracked: CPUTensor;
  } | null;
} {
  const chLength = x.shape[params.axis];
  const chStride = x.strides[params.axis];
  const innerLength = arrayProd(x.shape.slice(params.axis + 1));
  const innerStride = 1;
  const outerLength = arrayProd(x.shape.slice(0, params.axis));
  const outerStride = x.strides[Math.max(0, params.axis - 1)];
  const reduceLength = innerLength * outerLength;

  const xd = x.buffer.data;
  let updatedRunningStats: {
    runningMean: CPUTensor;
    runningVar: CPUTensor;
    numBatchesTracked: CPUTensor;
  } | null = null;
  // channel, [mean, var]
  const stats = CPUTensor.zeros([chLength, 2]);
  const sd = stats.buffer.data;
  // TODO: trainingフラグの意味をはっきりさせる
  if (params.training || !runningStats) {
    // calc stats
    for (let ch = 0; ch < chLength; ch++) {
      let sum = 0.0,
        sqsum = 0.0;
      for (let outer = 0; outer < outerLength; outer++) {
        for (let inner = 0; inner < innerLength; inner++) {
          const v =
            xd[ch * chStride + outer * outerStride + inner * innerStride];
          sum += v;
          sqsum += v * v;
        }
      }
      const mean = sum / reduceLength;
      const variance = sqsum / reduceLength - mean * mean;
      sd[ch * 2] = mean;
      sd[ch * 2 + 1] = variance;
    }

    if (params.trackRunningStats) {
      if (runningStats) {
        // update running stats
        const rmd = runningStats.runningMean.buffer.data;
        const rvd = runningStats.runningVar.buffer.data;
        const nbt = runningStats.numBatchesTracked.buffer.data[0];
        const updatedRunningMean = CPUTensor.zeros([chLength]);
        const updatedRunningVar = CPUTensor.zeros([chLength]);
        const updatedNumBatchesTracked = CPUTensor.fromArray(
          [nbt + 1],
          [],
          'int32'
        );
        const urmd = updatedRunningMean.buffer.data;
        const urvd = updatedRunningVar.buffer.data;
        const momentum =
          params.momentum != null ? params.momentum : 1 / nbt + 1;
        for (let ch = 0; ch < chLength; ch++) {
          // 正規化には標本分散を使用するが、runningVarには不偏分散を保存する(PyTorchと挙動を合わせる)
          urmd[ch] = (1.0 - momentum) * rmd[ch] + momentum * sd[ch * 2];
          urvd[ch] =
            (1.0 - momentum) * rvd[ch] +
            momentum * ((sd[ch * 2 + 1] * reduceLength) / (reduceLength - 1));
        }
        updatedRunningStats = {
          runningMean: updatedRunningMean,
          runningVar: updatedRunningVar,
          numBatchesTracked: updatedNumBatchesTracked,
        };
      } else {
        // make new running stats
        const updatedRunningMean = CPUTensor.zeros([chLength]);
        const updatedRunningVar = CPUTensor.zeros([chLength]);
        const updatedNumBatchesTracked = CPUTensor.fromArray([1], [], 'int32');
        const urmd = updatedRunningMean.buffer.data;
        const urvd = updatedRunningVar.buffer.data;
        const momentum = params.momentum != null ? params.momentum : 1;
        // runningMeanの初期値=0, runningVarの初期値=1
        for (let ch = 0; ch < chLength; ch++) {
          urmd[ch] = momentum * sd[ch * 2];
          urvd[ch] =
            1.0 -
            momentum +
            momentum * ((sd[ch * 2 + 1] * reduceLength) / (reduceLength - 1));
        }
        updatedRunningStats = {
          runningMean: updatedRunningMean,
          runningVar: updatedRunningVar,
          numBatchesTracked: updatedNumBatchesTracked,
        };
      }
    }
  } else {
    // copy stats from runningStats
    if (!runningStats) {
      throw new Error('batch_norm: training == false && runningStats == false');
    }

    const rmd = runningStats.runningMean.buffer.data;
    const rvd = runningStats.runningVar.buffer.data;
    for (let ch = 0; ch < chLength; ch++) {
      sd[ch * 2] = rmd[ch];
      sd[ch * 2 + 1] = rvd[ch];
    }
  }

  // computing scaling with affine
  // channel, [mean, invStd, scale, bias]
  const scalings = CPUTensor.zeros([chLength, 4]);
  const scd = scalings.buffer.data;
  const eps = params.eps;
  if (affine) {
    const wd = affine.weight.buffer.data;
    const bd = affine.bias.buffer.data;
    // (x - mean) * invStd * weight + bias
    // => invStd * weight, -mean * invStd * weight + bias
    for (let ch = 0; ch < chLength; ch++) {
      const mean = sd[ch * 2];
      const invStd = 1 / Math.sqrt(sd[ch * 2 + 1] + eps);
      scd[ch * 4] = mean;
      scd[ch * 4 + 1] = invStd;
      scd[ch * 4 + 2] = invStd * wd[ch];
      scd[ch * 4 + 3] = -mean * invStd * wd[ch] + bd[ch];
    }
  } else {
    // weight = 1, bias = 0
    for (let ch = 0; ch < chLength; ch++) {
      const mean = sd[ch * 2];
      const invStd = 1 / Math.sqrt(sd[ch * 2 + 1] + eps);
      scd[ch * 4] = mean;
      scd[ch * 4 + 1] = invStd;
      scd[ch * 4 + 2] = invStd;
      scd[ch * 4 + 3] = -mean * invStd;
    }
  }

  // compute output
  const y = CPUTensor.zeros(x.shape);
  const yd = y.buffer.data;
  for (let ch = 0; ch < chLength; ch++) {
    const scale = scd[ch * 4 + 2];
    const offset = scd[ch * 4 + 3];
    for (let outer = 0; outer < outerLength; outer++) {
      for (let inner = 0; inner < innerLength; inner++) {
        const v = xd[ch * chStride + outer * outerStride + inner * innerStride];
        const s = v * scale + offset;
        yd[ch * chStride + outer * outerStride + inner * innerStride] = s;
      }
    }
  }

  return { y, statsForBackprop: scalings, updatedRunningStats };
}

export function batch_norm_backprop_cpu(
  x: CPUTensor,
  gy: CPUTensor,
  statsForBackprop: CPUTensor,
  axis: number
): {
  gx: CPUTensor;
  gweight: CPUTensor;
  gbias: CPUTensor;
} {
  // TODO: 高速化
  const axesExceptCh = arange(gy.ndim);
  axesExceptCh.splice(axis, 1);

  const gbias = sum(gy, axesExceptCh, true);
  const chLength = gy.shape[axis];
  const reshapeShape = Array(x.ndim).fill(1);
  reshapeShape[axis] = chLength; // e.g. [1, chLength, 1, 1] for 2d image

  const mean = statsForBackprop.gets(slice(), 0).reshape(reshapeShape);
  const invStd = statsForBackprop.gets(slice(), 1).reshape(reshapeShape);
  const scale = statsForBackprop.gets(slice(), 2).reshape(reshapeShape);
  const gweight = CPUTensor.sub(
    CPUTensor.mul(sum(CPUTensor.mul(x, gy), axesExceptCh, true), invStd),
    CPUTensor.mul(gbias, CPUTensor.mul(mean, invStd))
  );
  const tmp = CPUTensor.mul(
    CPUTensor.add(
      CPUTensor.mul(CPUTensor.sub(x, mean), CPUTensor.mul(invStd, gweight)),
      gbias
    ),
    CPUTensor.s(1.0 / (gy.size / gbias.size))
  );
  const gx = CPUTensor.mul(scale, CPUTensor.sub(gy, tmp));
  return { gx, gweight: gweight.reshape([-1]), gbias: gbias.reshape([-1]) };
}

import { tidySync } from '../../../tidy';
import { arange } from '../../../util';
import { stridedCopy } from '../core/copy';
import { cat } from '../core/manipulation';
import { sum } from '../core/reduction';
import { WebGPUTensor } from '../webgpuTensor';

export interface BatchNormParams {
  axis: number;
  training: boolean;
  eps: number;
  momentum?: number;
  trackRunningStats: boolean;
}

function assertFloat32(tensors: WebGPUTensor[], functionName: string): void {
  for (const tensor of tensors) {
    if (tensor.dtype !== 'float32') {
      throw new Error(`${functionName}: input tensor must be float32`);
    }
  }
}

/**
 * axis以外の軸をすべて含む配列。sumの縮約軸として使う。
 */
function axesExceptAxis(ndim: number, axis: number): number[] {
  const axes = arange(ndim);
  axes.splice(axis, 1);
  return axes;
}

/**
 * チャンネル方向のみ長さを持つ形状。ブロードキャストのために使う。
 * 例: 2次元画像なら [1, chLength, 1, 1]
 */
function makeChannelShape(
  ndim: number,
  axis: number,
  chLength: number
): number[] {
  const shape = new Array<number>(ndim).fill(1);
  shape[axis] = chLength;
  return shape;
}

/**
 * CPU/WebGL実装と異なり専用カーネルを持たず、汎用のテンソル演算を組み合わせて実装している。
 */
export function batch_norm_webgpu(
  x: WebGPUTensor,
  affine: { weight: WebGPUTensor; bias: WebGPUTensor } | null,
  runningStats: {
    runningMean: WebGPUTensor;
    runningVar: WebGPUTensor;
    numBatchesTracked: WebGPUTensor;
  } | null,
  params: BatchNormParams
): {
  y: WebGPUTensor;
  statsForBackprop: WebGPUTensor;
  updatedRunningStats: {
    runningMean: WebGPUTensor;
    runningVar: WebGPUTensor;
    numBatchesTracked: WebGPUTensor;
  } | null;
} {
  assertFloat32([x], 'batch_norm');
  if (affine) {
    assertFloat32([affine.weight, affine.bias], 'batch_norm');
  }
  if (runningStats) {
    assertFloat32(
      [runningStats.runningMean, runningStats.runningVar],
      'batch_norm'
    );
    if (runningStats.numBatchesTracked.dtype !== 'int32') {
      throw new Error(
        `batch_norm: runningStats.numBatchesTracked tensor must be int32`
      );
    }
  }
  const { axis } = params;
  const chLength = x.shape[axis];
  const reduceLength = x.size / chLength;
  const chShape = makeChannelShape(x.ndim, axis, chLength);
  const reduceAxes = axesExceptAxis(x.ndim, axis);
  const computeRunningStats = params.training && params.trackRunningStats;

  const results: WebGPUTensor[] = tidySync(() => {
    let mean: WebGPUTensor;
    let variance: WebGPUTensor;
    // TODO: trainingフラグの意味をはっきりさせる
    if (params.training || !runningStats) {
      mean = WebGPUTensor.div(sum(x, reduceAxes, true), reduceLength);
      const sqsum = sum(WebGPUTensor.mul(x, x), reduceAxes, true);
      variance = WebGPUTensor.sub(
        WebGPUTensor.div(sqsum, reduceLength),
        WebGPUTensor.mul(mean, mean)
      );
    } else {
      mean = runningStats.runningMean.reshape(chShape);
      variance = runningStats.runningVar.reshape(chShape);
    }

    const invStd = WebGPUTensor.div(
      1,
      WebGPUTensor.sqrt(WebGPUTensor.add(variance, params.eps))
    );
    // (x - mean) * invStd * weight + bias
    // => x * (invStd * weight) + (-mean * invStd * weight + bias)
    const negMeanInvStd = WebGPUTensor.mul(WebGPUTensor.neg(mean), invStd);
    let scale: WebGPUTensor;
    let offset: WebGPUTensor;
    if (affine) {
      const weight = affine.weight.reshape(chShape);
      const bias = affine.bias.reshape(chShape);
      scale = WebGPUTensor.mul(invStd, weight);
      offset = WebGPUTensor.add(WebGPUTensor.mul(negMeanInvStd, weight), bias);
    } else {
      // weight = 1, bias = 0
      scale = invStd.alias();
      offset = negMeanInvStd.alias();
    }

    const y = WebGPUTensor.add(WebGPUTensor.mul(x, scale), offset);
    // channel, [mean, invStd, scale, bias]
    const statsForBackprop = cat(
      [
        mean.reshape([chLength, 1]),
        invStd.reshape([chLength, 1]),
        scale.reshape([chLength, 1]),
        offset.reshape([chLength, 1]),
      ],
      1
    );

    if (!computeRunningStats) {
      return [y, statsForBackprop];
    }
    // 正規化には標本分散を使用するが、runningVarには不偏分散を保存する(PyTorchと挙動を合わせる)
    const unbiasedVar = WebGPUTensor.mul(
      variance.reshape([chLength]),
      reduceLength / (reduceLength - 1)
    );
    const meanFlat = mean.reshape([chLength]);
    if (runningStats) {
      if (params.momentum == null) {
        // momentumがnullの場合の係数はnumBatchesTrackedの値に依存するが、
        // WebGPUではGPU上の値を同期的に読み出せない
        throw new Error(
          'batch_norm: momentum must be given for the WebGPU backend'
        );
      }
      const momentum = params.momentum;
      return [
        y,
        statsForBackprop,
        WebGPUTensor.add(
          WebGPUTensor.mul(runningStats.runningMean, 1 - momentum),
          WebGPUTensor.mul(meanFlat, momentum)
        ),
        WebGPUTensor.add(
          WebGPUTensor.mul(runningStats.runningVar, 1 - momentum),
          WebGPUTensor.mul(unbiasedVar, momentum)
        ),
        WebGPUTensor.add(
          runningStats.numBatchesTracked,
          WebGPUTensor.fromArray([1], [], 'int32')
        ),
      ];
    }
    // runningMeanの初期値=0, runningVarの初期値=1
    const momentum = params.momentum != null ? params.momentum : 1;
    return [
      y,
      statsForBackprop,
      WebGPUTensor.mul(meanFlat, momentum),
      WebGPUTensor.add(WebGPUTensor.mul(unbiasedVar, momentum), 1 - momentum),
      WebGPUTensor.fromArray([1], [], 'int32'),
    ];
  });

  return {
    y: results[0],
    statsForBackprop: results[1],
    updatedRunningStats: computeRunningStats
      ? {
          runningMean: results[2],
          runningVar: results[3],
          numBatchesTracked: results[4],
        }
      : null,
  };
}

export function batch_norm_backprop_webgpu(
  x: WebGPUTensor,
  gy: WebGPUTensor,
  statsForBackprop: WebGPUTensor,
  axis: number
): {
  gx: WebGPUTensor;
  gweight: WebGPUTensor;
  gbias: WebGPUTensor;
} {
  // TODO: 高速化
  const reduceAxes = axesExceptAxis(gy.ndim, axis);
  const chLength = gy.shape[axis];
  const chShape = makeChannelShape(x.ndim, axis, chLength);
  // statsForBackpropは[chLength, 4]。列を取り出してチャンネル形状にする。
  // 長さ1の軸のストライドは結果に影響しないため0でよい。
  const statsStride = new Array<number>(x.ndim).fill(0);
  statsStride[axis] = 4;

  const [gx, gweight, gbias] = tidySync(() => {
    const gbias = sum(gy, reduceAxes, true);
    const mean = stridedCopy(statsForBackprop, chShape, statsStride, 0);
    const invStd = stridedCopy(statsForBackprop, chShape, statsStride, 1);
    const scale = stridedCopy(statsForBackprop, chShape, statsStride, 2);

    const gweight = WebGPUTensor.sub(
      WebGPUTensor.mul(sum(WebGPUTensor.mul(x, gy), reduceAxes, true), invStd),
      WebGPUTensor.mul(gbias, WebGPUTensor.mul(mean, invStd))
    );
    const tmp = WebGPUTensor.mul(
      WebGPUTensor.add(
        WebGPUTensor.mul(
          WebGPUTensor.sub(x, mean),
          WebGPUTensor.mul(invStd, gweight)
        ),
        gbias
      ),
      1.0 / (gy.size / gbias.size)
    );
    const gx = WebGPUTensor.mul(scale, WebGPUTensor.sub(gy, tmp));
    return [gx, gweight.reshape([-1]), gbias.reshape([-1])];
  });

  return { gx, gweight, gbias };
}

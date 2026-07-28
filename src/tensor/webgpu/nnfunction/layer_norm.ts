import { tidySync } from '../../../tidy';
import { arange, arrayProd } from '../../../util';
import { getStride } from '../../shapeUtil';
import { stridedCopy } from '../core/copy';
import { cat } from '../core/manipulation';
import { sum } from '../core/reduction';
import { WebGPUTensor } from '../webgpuTensor';

export interface LayerNormParams {
  normalizedShape: ReadonlyArray<number>;
  eps: number;
}

function assertFloat32(tensors: WebGPUTensor[], functionName: string): void {
  for (const tensor of tensors) {
    if (tensor.dtype !== 'float32') {
      throw new Error(`${functionName}: input tensor must be float32`);
    }
  }
}

/**
 * 正規化する軸を長さ1にした形状。ブロードキャストのために使う。
 */
function makeOuterShape(
  xShape: ReadonlyArray<number>,
  normDims: number
): number[] {
  return [
    ...xShape.slice(0, xShape.length - normDims),
    ...new Array<number>(normDims).fill(1),
  ];
}

/**
 * CPU実装と異なり専用カーネルを持たず、汎用のテンソル演算を組み合わせて実装している。
 */
export function layer_norm_webgpu(
  x: WebGPUTensor,
  affine: { weight: WebGPUTensor; bias: WebGPUTensor },
  params: LayerNormParams
): {
  y: WebGPUTensor;
  statsForBackprop: WebGPUTensor;
} {
  assertFloat32([x, affine.weight, affine.bias], 'layer_norm');
  const normDims = params.normalizedShape.length;
  const reduceLength = arrayProd(params.normalizedShape);
  const outerLength = x.size / reduceLength;
  const reduceAxes = arange(x.ndim - normDims, x.ndim);

  const [y, statsForBackprop] = tidySync(() => {
    const mean = WebGPUTensor.div(sum(x, reduceAxes, true), reduceLength);
    const sqsum = sum(WebGPUTensor.mul(x, x), reduceAxes, true);
    const variance = WebGPUTensor.sub(
      WebGPUTensor.div(sqsum, reduceLength),
      WebGPUTensor.mul(mean, mean)
    );
    const invStd = WebGPUTensor.div(
      1,
      WebGPUTensor.sqrt(WebGPUTensor.add(variance, params.eps))
    );
    const normalized = WebGPUTensor.mul(WebGPUTensor.sub(x, mean), invStd);
    const scaled = WebGPUTensor.add(
      WebGPUTensor.mul(normalized, affine.weight),
      affine.bias
    );
    // outerLength, [mean, invStd]
    const stats = cat(
      [mean.reshape([outerLength, 1]), invStd.reshape([outerLength, 1])],
      1
    );
    return [scaled, stats];
  });

  return { y, statsForBackprop };
}

export function layer_norm_backprop_webgpu(
  x: WebGPUTensor,
  w: WebGPUTensor,
  gy: WebGPUTensor,
  statsForBackprop: WebGPUTensor,
  params: LayerNormParams
): {
  gx: WebGPUTensor;
  gweight: WebGPUTensor;
  gbias: WebGPUTensor;
} {
  // TODO: 高速化
  const normDims = params.normalizedShape.length;
  const n = arrayProd(params.normalizedShape);
  const outerAxes = arange(gy.ndim - normDims);
  const normAxes = arange(gy.ndim - normDims, gy.ndim);
  const outerShape = makeOuterShape(x.shape, normDims);
  // statsForBackpropは[outerLength, 2]。列を取り出してouterShapeにする。
  // 長さ1の軸のストライドは結果に影響しないため0でよい。
  const outerStrides = getStride(x.shape.slice(0, x.ndim - normDims));
  const statsStride = new Array<number>(x.ndim).fill(0);
  for (let d = 0; d < outerStrides.length; d++) {
    statsStride[d] = outerStrides[d] * 2;
  }

  const [gx, gweight, gbias] = tidySync(() => {
    const gbias = sum(gy, outerAxes, true);
    const mean = stridedCopy(statsForBackprop, outerShape, statsStride, 0);
    const invStd = stridedCopy(statsForBackprop, outerShape, statsStride, 1);

    const gweight = WebGPUTensor.sub(
      sum(WebGPUTensor.mul(WebGPUTensor.mul(x, gy), invStd), outerAxes, true),
      sum(WebGPUTensor.mul(gy, WebGPUTensor.mul(mean, invStd)), outerAxes, true)
    );
    const xScaled = WebGPUTensor.mul(WebGPUTensor.sub(x, mean), invStd);
    const gxh = WebGPUTensor.mul(gy, w);
    const tmp = WebGPUTensor.sub(
      WebGPUTensor.sub(WebGPUTensor.mul(n, gxh), sum(gxh, normAxes, true)),
      WebGPUTensor.mul(
        xScaled,
        sum(WebGPUTensor.mul(gxh, xScaled), normAxes, true)
      )
    );
    const gx = WebGPUTensor.mul(WebGPUTensor.mul(1 / n, invStd), tmp);
    return [
      gx,
      gweight.reshape(params.normalizedShape),
      gbias.reshape(params.normalizedShape),
    ];
  });

  return { gx, gweight, gbias };
}

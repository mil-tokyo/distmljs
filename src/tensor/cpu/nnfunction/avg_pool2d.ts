import { avgPool2DCalcShape } from '../../nnFunctionUtil';
import { CPUTensor } from '../cpuTensor';

export function avg_pool2d_cpu(
  x: CPUTensor,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    ceilMode: boolean;
    countIncludePad: boolean;
    divisorOverride?: number;
  }
): CPUTensor {
  const {
    batch,
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0b, pads1b, pads0e, pads1e],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = avgPool2DCalcShape(params, x.shape);
  const output = CPUTensor.zeros([batch, ch, outShape0, outShape1]);
  const dX = x.getBuffer().data;
  const dI = output.getBuffer().data;
  let idx = 0;
  // PyTorchでの除算対象面積の挙動
  // ceilModeにより、paddingよりも外の領域がsliding windowに入るとき、countIncludePad==trueの場合でも面積に数えない
  let isConstantDiv = false;
  let multiplierConstant = 0.0;
  if (params.divisorOverride) {
    isConstantDiv = true;
    multiplierConstant = 1 / params.divisorOverride;
  }
  const countIncludePad = params.countIncludePad;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < ch; c++) {
      for (let oy = 0; oy < outShape0; oy++) {
        for (let ox = 0; ox < outShape1; ox++) {
          let mv = 0.0;
          let imageArea = 0;
          let padArea = 0;
          for (let ky = 0; ky < kernelShape0; ky++) {
            for (let kx = 0; kx < kernelShape1; kx++) {
              const iny = oy * strides0 - pads0b + ky,
                inx = ox * strides1 - pads1b + kx;
              if (iny >= 0 && iny < inShape0 && inx >= 0 && inx < inShape1) {
                const xidx = ((b * ch + c) * inShape0 + iny) * inShape1 + inx;
                const v = dX[xidx];
                mv += v;
                imageArea++;
              }
              if (iny < inShape0 + pads0e && inx < inShape1 + pads1e) {
                padArea++;
              }
            }
          }

          if (isConstantDiv) {
            mv *= multiplierConstant;
          } else if (countIncludePad) {
            mv /= padArea;
          } else {
            mv /= imageArea;
          }
          dI[idx++] = mv;
        }
      }
    }
  }
  return output;
}

export function avg_pool2d_backprop_cpu(
  gy: CPUTensor,
  xShape: ReadonlyArray<number>,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    ceilMode: boolean;
    countIncludePad: boolean;
    divisorOverride?: number;
  }
): CPUTensor {
  const {
    batch,
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0b, pads1b, pads0e, pads1e],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = avgPool2DCalcShape(params, xShape);
  // currently implements only global average pooling
  if (
    kernelShape0 === inShape0 &&
    kernelShape1 === inShape1 &&
    pads0b === 0 &&
    pads0e === 0 &&
    pads1b === 0 &&
    pads1e === 0 &&
    strides0 === kernelShape0 &&
    strides1 === kernelShape1 &&
    outShape0 === 1 &&
    outShape1 === 1
  ) {
    // global average pooling

    const gx = CPUTensor.zeros([batch, ch, inShape0, inShape1]);
    const dGy = gy.getBuffer().data;
    const dGx = gx.getBuffer().data;
    const inSpLen = inShape0 * inShape1;
    const outSpLen = outShape0 * outShape1;
    const mul = params.divisorOverride
      ? 1 / params.divisorOverride
      : 1 / inSpLen;
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < ch; c++) {
        const val = dGy[(b * ch + c) * outSpLen] * mul;
        for (let isp = 0; isp < inSpLen; isp++) {
          dGx[(b * ch + c) * inSpLen + isp] = val;
        }
      }
    }

    return gx;
  } else {
    throw new Error(
      'currently, avg_pool2d_backprop_cpu only implements global average pooling'
    );
  }
}

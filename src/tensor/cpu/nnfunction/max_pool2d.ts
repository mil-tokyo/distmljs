import { maxPool2DCalcShape } from '../../nnFunctionUtil';
import { CPUTensor } from '../cpuTensor';

export function max_pool2d_cpu(
  x: CPUTensor,
  params: {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    returnIndices: false;
    ceilMode: boolean;
  }
): CPUTensor {
  const {
    batch,
    dilations: [dilations0, dilations1],
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0, pads1],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = maxPool2DCalcShape(params, x.shape);
  const output = CPUTensor.zeros([batch, ch, outShape0, outShape1]);
  const dX = x.getBuffer().data;
  const dI = output.getBuffer().data;
  let idx = 0;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < ch; c++) {
      for (let oy = 0; oy < outShape0; oy++) {
        for (let ox = 0; ox < outShape1; ox++) {
          let mv = -Infinity;
          for (let ky = 0; ky < kernelShape0; ky++) {
            for (let kx = 0; kx < kernelShape1; kx++) {
              const iny = oy * strides0 - pads0 + ky * dilations0,
                inx = ox * strides1 - pads1 + kx * dilations1;
              if (iny >= 0 && iny < inShape0 && inx >= 0 && inx < inShape1) {
                const xidx = ((b * ch + c) * inShape0 + iny) * inShape1 + inx,
                  v = dX[xidx];
                if (v > mv) {
                  mv = v;
                  // Max position: xidxを出力
                }
              }
            }
          }

          dI[idx++] = mv;
        }
      }
    }
  }
  return output;
}

export function max_pool2d_with_indices_cpu(
  x: CPUTensor,
  params: {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    returnIndices: true;
    ceilMode: boolean;
  }
): CPUTensor[] {
  throw new Error('not implemented');
}

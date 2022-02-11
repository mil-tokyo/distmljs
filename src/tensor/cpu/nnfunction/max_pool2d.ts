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
    returnIndices: true | 'spatial' | 'flatten';
    ceilMode: boolean;
  }
): CPUTensor[] {
  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
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
  const oIdx = CPUTensor.zeros([batch, ch, outShape0, outShape1], 'int32');
  const dX = x.getBuffer().data;
  const dI = output.getBuffer().data;
  const dD = oIdx.getBuffer().data;
  let idx = 0;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < ch; c++) {
      for (let oy = 0; oy < outShape0; oy++) {
        for (let ox = 0; ox < outShape1; ox++) {
          let mv = -Infinity;
          let maxPos = 0;
          for (let ky = 0; ky < kernelShape0; ky++) {
            for (let kx = 0; kx < kernelShape1; kx++) {
              const iny = oy * strides0 - pads0 + ky * dilations0,
                inx = ox * strides1 - pads1 + kx * dilations1;
              if (iny >= 0 && iny < inShape0 && inx >= 0 && inx < inShape1) {
                const xidx = ((b * ch + c) * inShape0 + iny) * inShape1 + inx,
                  v = dX[xidx];
                if (v > mv) {
                  mv = v;
                  // flattenモードの場合はmaxPos==xidx
                  maxPos = iny * inShape1 + inx;
                }
              }
            }
          }

          dD[idx] = maxPos;
          dI[idx++] = mv;
        }
      }
    }
  }
  return [output, oIdx];
}

export function max_pool2d_backprop_cpu(
  indices: CPUTensor,
  gy: CPUTensor,
  xShape: ReadonlyArray<number>,
  params: {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    ceilMode: boolean;
    returnIndices: true | 'spatial' | 'flatten';
  }
): CPUTensor {
  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
  const gx = CPUTensor.zeros(xShape);
  const dGx = gx.getBuffer().data;
  const dI = indices.getBuffer().data;
  const dGy = gy.getBuffer().data;

  const [batch, ch, inShape0, inShape1] = xShape;
  const inSpLen = inShape0 * inShape1;
  const [, , outShape0, outShape1] = indices.shape;
  const outSpLen = outShape0 * outShape1;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < ch; c++) {
      for (let os = 0; os < outSpLen; os++) {
        const spIdx = dI[(b * ch + c) * outSpLen + os];
        const gyv = dGy[(b * ch + c) * outSpLen + os];
        // stride < kernelSize の場合、x.gradの同じ要素に複数の勾配が足し合わさる場合がある
        // そのため = ではなく += を使用
        dGx[(b * ch + c) * inSpLen + spIdx] += gyv;
      }
    }
  }
  return gx;
}

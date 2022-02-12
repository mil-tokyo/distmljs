import { conv2DCalcShape } from '../../nnFunctionUtil';
import { bmm } from '../core/gemm';
import { CPUTensor } from '../cpuTensor';

interface Conv2dImplParams {
  stride: number | [number, number];
  padding: number | [number, number] | [number, number, number, number]; //TODO: support 'same' and 'valid'
  dilation: number | [number, number];
  groups: number;
}

export function conv2d_cpu(
  x: CPUTensor,
  weight: CPUTensor,
  bias: CPUTensor | undefined,
  params: Conv2dImplParams
): CPUTensor {
  const {
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chIn,
    chInPerGroup,
    chOut,
    chOutPerGroup,
  } = conv2DCalcShape(params, x.shape, weight.shape);
  // TODO im2colが巨大になる場合に分割して実行
  const im2colData = CPUTensor.zeros([
    group,
    batch,
    outShape[0],
    outShape[1],
    chInPerGroup,
    kernelShape[0],
    kernelShape[1],
  ]);
  im2col(
    x.buffer.data as Float32Array,
    im2colData.buffer.data as Float32Array,
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chIn,
    chInPerGroup
  );

  // dI(group, bout, cinkhkw) * dW(group, coutpergroup, cinkhkw) -> dT(group, bout, coutpergroup)
  const matmulDataRs = bmm(
    im2colData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]),
    weight.reshape([
      group,
      chOutPerGroup,
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]),
    false,
    true
  );
  im2colData.dispose();
  const matmulData = matmulDataRs.reshape([
    group,
    batch,
    outShape[0],
    outShape[1],
    chOutPerGroup,
  ]);
  const yRs = matmulData.transpose([1, 0, 4, 2, 3]);
  matmulData.dispose();
  const y = yRs.reshape([batch, chOut, outShape[0], outShape[1]]);
  if (bias) {
    return CPUTensor.add(y, bias.reshape([1, -1, 1, 1]));
  } else {
    return y;
  }
}

export function conv2d_backprop_gb_cpu(gy: CPUTensor): CPUTensor {
  return CPUTensor.sum(gy, [0, 2, 3]);
}

// TODO: extend to conv_transpose2d with minor change
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  weight: CPUTensor,
  skipGx: true,
  skipGw: false,
  params: Conv2dImplParams
): [null, CPUTensor];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  weight: CPUTensor,
  skipGx: false,
  skipGw: true,
  params: Conv2dImplParams
): [CPUTensor, null];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  weight: CPUTensor,
  skipGx: false,
  skipGw: false,
  params: Conv2dImplParams
): [CPUTensor, CPUTensor];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  weight: CPUTensor,
  skipGx: boolean,
  skipGw: boolean,
  params: Conv2dImplParams
): [CPUTensor | null, CPUTensor | null] {
  const {
    batch,
    dilations,
    group,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    chIn,
    chInPerGroup,
    chOut,
    chOutPerGroup,
  } = conv2DCalcShape(params, x.shape, weight.shape);
  // TODO im2colが巨大になる場合に分割して実行
  const gyg = gy.reshape([
    batch,
    group,
    chOutPerGroup,
    outShape[0],
    outShape[1],
  ]);
  const gyTransposeData = gyg.transpose([1, 0, 3, 4, 2]);
  let gw: CPUTensor | null = null;
  let gx: CPUTensor | null = null;
  if (!skipGw) {
    const im2colData = CPUTensor.zeros([
      group,
      batch,
      outShape[0],
      outShape[1],
      chInPerGroup,
      kernelShape[0],
      kernelShape[1],
    ]);
    im2col(
      x.buffer.data as Float32Array,
      im2colData.buffer.data as Float32Array,
      batch,
      dilations,
      group,
      kernelShape,
      pads,
      strides,
      inShape,
      outShape,
      chIn,
      chInPerGroup
    );
    // dI(group, bout, cinkhkw) * dGyT(group, bout, coutpergroup) -> dgw(group, coutpergroup, cinkhkw)
    const im2colDataRs = im2colData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    im2colData.dispose();

    const gyTRs = gyTransposeData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);

    const gwRs = bmm(gyTRs, im2colDataRs, true, false);
    gyTRs.dispose();
    im2colDataRs.dispose();
    gw = gwRs.reshape([chOut, chInPerGroup, kernelShape[0], kernelShape[1]]);
    gwRs.dispose();
  }
  if (!skipGx) {
    // dGyT(group, bout, coutpergroup) * dW(group, coutpergroup, cinkhkw) -> dGi(group, bout, cinkhkw)
    const gyTRs = gyTransposeData.reshape([
      group,
      batch * outShape[0] * outShape[1],
      chOutPerGroup,
    ]);
    const weightRs = weight.reshape([
      group,
      chOutPerGroup,
      chInPerGroup * kernelShape[0] * kernelShape[1],
    ]);
    const matmul = bmm(gyTRs, weightRs, false, false);
    gyTRs.dispose();
    weightRs.dispose();
    gx = CPUTensor.zeros([batch, chIn, inShape[0], inShape[1]]);
    col2im(
      matmul.buffer.data as Float32Array,
      gx.buffer.data as Float32Array,
      batch,
      dilations,
      group,
      kernelShape,
      pads,
      strides,
      outShape,
      inShape,
      chInPerGroup
    );
  }
  return [gx, gw];
}

function im2col(
  dX: Float32Array,
  dI: Float32Array,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  inShape: number[],
  outShape: number[],
  chIn: number,
  chInPerGroup: number
): void {
  const [inShape0, inShape1] = inShape;
  const [kernelShape0, kernelShape1] = kernelShape;
  const [pads0, pads1] = pads;
  const [dilations0, dilations1] = dilations;
  const [strides0, strides1] = strides;
  const [outShape0, outShape1] = outShape;
  let idx = 0;
  for (let g = 0; g < group; g++) {
    for (let b = 0; b < batch; b++) {
      for (let oy = 0; oy < outShape0; oy++) {
        for (let ox = 0; ox < outShape1; ox++) {
          for (let ci = 0; ci < chInPerGroup; ci++) {
            for (let ky = 0; ky < kernelShape0; ky++) {
              for (let kx = 0; kx < kernelShape1; kx++) {
                let v = 0;
                const iny = oy * strides0 - pads0 + ky * dilations0,
                  inx = ox * strides1 - pads1 + kx * dilations1;
                if (iny >= 0 && iny < inShape0 && inx >= 0 && inx < inShape1) {
                  v =
                    dX[
                      ((b * chIn + g * chInPerGroup + ci) * inShape0 + iny) *
                        inShape1 +
                        inx
                    ];
                }
                dI[idx++] = v;
              }
            }
          }
        }
      }
    }
  }
}

function col2im(
  dI: Float32Array,
  dY: Float32Array,
  batch: number,
  dilations: number[],
  group: number,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  outShape: number[],
  inShape: number[],
  chInPerGroup: number
): void {
  let idx = 0;
  // in/outはconvのforward基準 (convtransposeとは逆)
  // dI: group, batch, outShape[0], outShape[1], chInPerGroup, kernelShape[0], kernelShape[1]
  // dY: batch, group, chInPerGroup, inShape[0], inShape[1]
  const [inShape0, inShape1] = inShape;
  const [kernelShape0, kernelShape1] = kernelShape;
  const [pads0, pads1] = pads;
  const [dilations0, dilations1] = dilations;
  const [strides0, strides1] = strides;
  const [outShape0, outShape1] = outShape;
  for (let b = 0; b < batch; b++) {
    for (let g = 0; g < group; g++) {
      for (let co = 0; co < chInPerGroup; co++) {
        for (let o0 = 0; o0 < inShape0; o0++) {
          for (let o1 = 0; o1 < inShape1; o1++) {
            let v = 0;
            for (let k0 = 0; k0 < kernelShape0; k0++) {
              for (let k1 = 0; k1 < kernelShape1; k1++) {
                const i0s = o0 + pads0 - k0 * dilations0;
                const i1s = o1 + pads1 - k1 * dilations1;
                if (i0s % strides0 !== 0 || i1s % strides1 !== 0) {
                  continue;
                }

                const i0 = i0s / strides0;
                const i1 = i1s / strides1;
                if (i0 < 0 || i0 >= outShape0 || i1 < 0 || i1 >= outShape1) {
                  continue;
                }
                v +=
                  dI[
                    (((((g * batch + b) * outShape0 + i0) * outShape1 + i1) *
                      chInPerGroup +
                      co) *
                      kernelShape0 +
                      k0) *
                      kernelShape1 +
                      k1
                  ];
              }
            }
            dY[idx++] = v;
          }
        }
      }
    }
  }
}

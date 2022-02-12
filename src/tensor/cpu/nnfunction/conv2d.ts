import { conv2DCalcShape } from '../../nnFunctionUtil';
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
  const im2colNumel =
    group *
    batch *
    outShape[0] *
    outShape[1] *
    chInPerGroup *
    kernelShape[0] *
    kernelShape[1];
  // TODO im2colが巨大になる場合に分割して実行
  const im2colData = new Float32Array(im2colNumel),
    matmulData = new Float32Array(
      group * batch * outShape[0] * outShape[1] * chOutPerGroup
    );
  im2col(
    x.buffer.data as Float32Array,
    im2colData,
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
  matmul_forward(
    im2colData,
    weight.buffer.data as Float32Array,
    matmulData,
    group,
    batch * outShape[0] * outShape[1],
    chInPerGroup * kernelShape[0] * kernelShape[1],
    chOutPerGroup
  );
  const y = CPUTensor.zeros([batch, chOut, outShape[0], outShape[1]]);
  transpose_forward_y(
    matmulData,
    y.buffer.data as Float32Array,
    group,
    batch,
    outShape[0] * outShape[1],
    chOutPerGroup
  );
  if (bias) {
    addBias(
      bias.buffer.data as Float32Array,
      y.buffer.data as Float32Array,
      batch,
      chOut,
      outShape[0] * outShape[1]
    );
  }

  return y;
}

export function conv2d_backprop_gb_cpu(gy: CPUTensor): CPUTensor {
  const [batch, chOut, outShape0, outShape1] = gy.shape;
  const gb = CPUTensor.zeros([chOut]);
  reduceBias(
    gy.buffer.data as Float32Array,
    gb.buffer.data as Float32Array,
    batch,
    chOut,
    outShape0 * outShape1
  );
  return gb;
}

// TODO: extend to conv_transpose2d with minor change
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  w: CPUTensor,
  skipGx: true,
  skipGw: false,
  params: Conv2dImplParams
): [null, CPUTensor];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  w: CPUTensor,
  skipGx: false,
  skipGw: true,
  params: Conv2dImplParams
): [CPUTensor, null];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  w: CPUTensor,
  skipGx: false,
  skipGw: false,
  params: Conv2dImplParams
): [CPUTensor, CPUTensor];
export function conv2d_backprop_gxgw_cpu(
  gy: CPUTensor,
  x: CPUTensor,
  w: CPUTensor,
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
  } = conv2DCalcShape(params, x.shape, w.shape);
  const im2colNumel =
    group *
    batch *
    outShape[0] *
    outShape[1] *
    chInPerGroup *
    kernelShape[0] *
    kernelShape[1];
  // TODO im2colが巨大になる場合に分割して実行
  const gyTransposeData = new Float32Array(
    group * batch * outShape[0] * outShape[1] * chOutPerGroup
  );
  transpose_gy_gw(
    gy.buffer.data as Float32Array,
    gyTransposeData,
    group,
    batch,
    outShape[0] * outShape[1],
    chOutPerGroup
  );
  let gw: CPUTensor | null = null;
  let gx: CPUTensor | null = null;
  if (!skipGw) {
    const im2colData = new Float32Array(im2colNumel);
    im2col(
      x.buffer.data as Float32Array,
      im2colData,
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
    gw = CPUTensor.zeros([chOut, chInPerGroup, kernelShape[0], kernelShape[1]]);
    matmul_gw(
      im2colData,
      gyTransposeData,
      gw.buffer.data as Float32Array,
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
      chOutPerGroup
    );
  }
  if (!skipGx) {
    const matmulData = new Float32Array(
      group *
        batch *
        outShape[0] *
        outShape[1] *
        chInPerGroup *
        kernelShape[0] *
        kernelShape[1]
    );
    matmul_gx(
      gyTransposeData,
      w.buffer.data as Float32Array,
      matmulData,
      group,
      batch * outShape[0] * outShape[1],
      chInPerGroup * kernelShape[0] * kernelShape[1],
      chOutPerGroup
    );
    gx = CPUTensor.zeros([batch, chIn, inShape[0], inShape[1]]);
    col2im(
      matmulData,
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
  let idx = 0;
  for (let g = 0; g < group; g++) {
    for (let b = 0; b < batch; b++) {
      for (let oy = 0; oy < outShape[0]; oy++) {
        for (let ox = 0; ox < outShape[1]; ox++) {
          for (let ci = 0; ci < chInPerGroup; ci++) {
            for (let ky = 0; ky < kernelShape[0]; ky++) {
              for (let kx = 0; kx < kernelShape[1]; kx++) {
                let v = 0;
                const iny = oy * strides[0] - pads[0] + ky * dilations[0],
                  inx = ox * strides[1] - pads[1] + kx * dilations[1];
                if (
                  iny >= 0 &&
                  iny < inShape[0] &&
                  inx >= 0 &&
                  inx < inShape[1]
                ) {
                  v =
                    dX[
                      ((b * chIn + g * chInPerGroup + ci) * inShape[0] + iny) *
                        inShape[1] +
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
  for (let b = 0; b < batch; b++) {
    for (let g = 0; g < group; g++) {
      for (let co = 0; co < chInPerGroup; co++) {
        for (let o0 = 0; o0 < inShape[0]; o0++) {
          for (let o1 = 0; o1 < inShape[1]; o1++) {
            let v = 0;
            for (let k0 = 0; k0 < kernelShape[0]; k0++) {
              for (let k1 = 0; k1 < kernelShape[1]; k1++) {
                const i0s = o0 + pads[0] - k0 * dilations[0];
                const i1s = o1 + pads[1] - k1 * dilations[1];
                if (i0s % strides[0] !== 0 || i1s % strides[1] !== 0) {
                  continue;
                }

                const i0 = i0s / strides[0];
                const i1 = i1s / strides[1];
                if (
                  i0 < 0 ||
                  i0 >= outShape[0] ||
                  i1 < 0 ||
                  i1 >= outShape[1]
                ) {
                  continue;
                }
                v +=
                  dI[
                    (((((g * batch + b) * outShape[0] + i0) * outShape[1] +
                      i1) *
                      chInPerGroup +
                      co) *
                      kernelShape[0] +
                      k0) *
                      kernelShape[1] +
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

function matmul_forward(
  dI: Float32Array,
  dW: Float32Array,
  dT: Float32Array,
  group: number,
  bout: number,
  cinkhkw: number,
  chOutPerGroup: number
) {
  // dI(group, bout, cinkhkw) * dW(group, coutpergroup, cinkhkw) -> dT(group, bout, coutpergroup)
  for (let g = 0; g < group; g++) {
    for (let y = 0; y < bout; y++) {
      for (let x = 0; x < chOutPerGroup; x++) {
        let s = 0;
        for (let ip = 0; ip < cinkhkw; ip++) {
          s +=
            dI[(g * bout + y) * cinkhkw + ip] *
            dW[(g * chOutPerGroup + x) * cinkhkw + ip];
        }
        dT[(g * bout + y) * chOutPerGroup + x] = s;
      }
    }
  }
}

function matmul_gw(
  dI: Float32Array,
  dGyT: Float32Array,
  dGw: Float32Array,
  group: number,
  bout: number,
  cinkhkw: number,
  chOutPerGroup: number
) {
  // dI(group, bout, cinkhkw) * dGyT(group, bout, coutpergroup) -> dgw(group, coutpergroup, cinkhkw)
  for (let g = 0; g < group; g++) {
    for (let y = 0; y < chOutPerGroup; y++) {
      for (let x = 0; x < cinkhkw; x++) {
        let s = 0;
        for (let ip = 0; ip < bout; ip++) {
          s +=
            dI[(g * bout + ip) * cinkhkw + x] *
            dGyT[(g * bout + ip) * chOutPerGroup + y];
        }
        dGw[(g * chOutPerGroup + y) * cinkhkw + x] = s;
      }
    }
  }
}

function matmul_gx(
  dGyT: Float32Array,
  dW: Float32Array,
  dGi: Float32Array,
  group: number,
  bout: number,
  cinkhkw: number,
  chOutPerGroup: number
) {
  // dGyT(group, bout, coutpergroup) * dW(group, coutpergroup, cinkhkw) -> dGi(group, bout, cinkhkw)
  for (let g = 0; g < group; g++) {
    for (let y = 0; y < bout; y++) {
      for (let x = 0; x < cinkhkw; x++) {
        let s = 0;
        for (let ip = 0; ip < chOutPerGroup; ip++) {
          s +=
            dGyT[(g * bout + y) * chOutPerGroup + ip] *
            dW[(g * chOutPerGroup + ip) * cinkhkw + x];
        }
        dGi[(g * bout + y) * cinkhkw + x] = s;
      }
    }
  }
}
function transpose_forward_y(
  dT: Float32Array,
  dO: Float32Array,
  group: number,
  batch: number,
  outarea: number,
  chOutPerGroup: number
) {
  // dT(group, batch, outh, outw, choutpergroup) -> dO(batch, group, choutpergroup, outh, outw)
  let idx = 0;
  for (let b = 0; b < batch; b++) {
    for (let g = 0; g < group; g++) {
      for (let c = 0; c < chOutPerGroup; c++) {
        for (let x = 0; x < outarea; x++) {
          dO[idx++] = dT[((g * batch + b) * outarea + x) * chOutPerGroup + c];
        }
      }
    }
  }
}

function transpose_gy_gw(
  dGy: Float32Array,
  dGyT: Float32Array,
  group: number,
  batch: number,
  outarea: number,
  chOutPerGroup: number
) {
  // dGy(batch, group, choutpergroup, outh, outw) -> dGyT(group, batch, outh, outw, choutpergroup)
  let idx = 0;
  for (let g = 0; g < group; g++) {
    for (let b = 0; b < batch; b++) {
      for (let x = 0; x < outarea; x++) {
        for (let c = 0; c < chOutPerGroup; c++) {
          dGyT[idx++] =
            dGy[((b * group + g) * chOutPerGroup + c) * outarea + x];
        }
      }
    }
  }
}

function addBias(
  dB: Float32Array,
  dO: Float32Array,
  batch: number,
  chOut: number,
  outarea: number
) {
  let idx = 0;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < chOut; c++) {
      for (let x = 0; x < outarea; x++) {
        dO[idx++] += dB[c];
      }
    }
  }
}

function reduceBias(
  dGy: Float32Array,
  dGb: Float32Array,
  batch: number,
  chOut: number,
  outarea: number
) {
  let idx = 0;
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < chOut; c++) {
      for (let x = 0; x < outarea; x++) {
        dGb[c] += dGy[idx++];
      }
    }
  }
}

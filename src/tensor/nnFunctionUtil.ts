interface MaxPoolParams {
  autoPad?: 'NOTSET' | 'SAME_UPPER' | 'SAME_LOWER' | 'VALID'; //for ONNX
  kernelSize: number | number[];
  stride: number | number[];
  padding: number | number[];
  dilation: number | number[];
  ceilMode: boolean;
}

function getDimOrScalar(
  v: number | number[],
  ndim: number,
  idx: number,
  top: boolean
): number {
  if (typeof v === 'number') {
    return v;
  } else {
    // ndim==2の場合、[y, x]と[ybegin, xbegin, yend, xend]の2通りを想定する
    let i: number;
    if (v.length === ndim) {
      i = idx;
    } else if (v.length === ndim * 2) {
      i = idx * 2 + (top ? 0 : 1);
    } else {
      throw new Error(`length of v must be ${ndim} or ${ndim * 2}`);
    }
    const x = v[i];
    if (typeof x === 'number') {
      return x;
    } else {
      throw new Error('v is not number or number[]');
    }
  }
}

function maxPoolCalcAxisShape(
  params: MaxPoolParams,
  ndim: number,
  dimIdx: number,
  width: number
) {
  const kernelSize = getDimOrScalar(params.kernelSize, ndim, dimIdx, true);
  const dilation = getDimOrScalar(params.dilation, ndim, dimIdx, true);
  let padding0 = getDimOrScalar(params.padding, ndim, dimIdx, true);
  let padding1 = getDimOrScalar(params.padding, ndim, dimIdx, false);
  const stride = getDimOrScalar(params.stride, ndim, dimIdx, true);

  let shape: number;
  if (!params.autoPad || params.autoPad === 'NOTSET') {
    if (params.ceilMode) {
      // ceilModeにおいて、右端のpaddingしか含まれないウィンドウは除外する
      shape =
        Math.ceil(
          (width + padding0 - dilation * (kernelSize - 1) - 1) / stride
        ) + 1;
    } else {
      shape =
        Math.floor(
          (width + padding0 + padding1 - dilation * (kernelSize - 1) - 1) /
            stride
        ) + 1;
    }
  } else if (
    params.autoPad === 'SAME_LOWER' ||
    params.autoPad === 'SAME_UPPER'
  ) {
    // calculate output shape as if padding is zero
    shape = Math.ceil(width / stride);
    const sumPad =
      (shape - 1) * stride + ((kernelSize - 1) * dilation + 1) - width;
    if (params.autoPad === 'SAME_LOWER') {
      padding0 = Math.ceil(sumPad / 2);
      padding1 = Math.floor(sumPad / 2);
    } else {
      padding0 = Math.floor(sumPad / 2);
      padding1 = Math.ceil(sumPad / 2);
    }
  } else if (params.autoPad === 'VALID') {
    shape = Math.ceil((width - dilation * (kernelSize - 1)) / stride);
    padding0 = padding1 = 0;
  } else {
    throw new Error(`Unknown autoPad ${params.autoPad}`);
  }

  return { shape, kernelSize, padding0, padding1, dilation, stride };
}

export function maxPool2DCalcShape(
  params: MaxPoolParams,
  dimsX: ReadonlyArray<number>
) {
  const [batch, ch] = dimsX;
  const inShape: number[] = [];
  const dilations: number[] = [];
  const kernelShape: number[] = [];
  const pads: number[] = [0, 0, 0, 0]; // [ybegin, xbegin, yend, xend]
  const strides: number[] = [];
  const outShape: number[] = [];
  for (let dim = 0; dim < 2; dim++) {
    const width = dimsX[2 + dim];
    const { shape, kernelSize, padding0, padding1, dilation, stride } =
      maxPoolCalcAxisShape(params, 2, dim, width);
    inShape.push(width);
    dilations.push(dilation);
    kernelShape.push(kernelSize);
    pads[dim] = padding0;
    pads[dim + 2] = padding1;
    strides.push(stride);
    outShape.push(shape);
  }

  return {
    batch,
    dilations,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    ch,
  };
}

interface AvgPoolParams {
  autoPad?: 'NOTSET' | 'SAME_UPPER' | 'SAME_LOWER' | 'VALID'; //for ONNX
  kernelSize: number | number[];
  stride: number | number[];
  padding: number | number[];
  ceilMode: boolean;
}

function avgPoolCalcAxisShape(
  params: AvgPoolParams,
  ndim: number,
  dimIdx: number,
  width: number
) {
  const kernelSize = getDimOrScalar(params.kernelSize, ndim, dimIdx, true);
  let padding0 = getDimOrScalar(params.padding, ndim, dimIdx, true);
  let padding1 = getDimOrScalar(params.padding, ndim, dimIdx, false);
  const stride = getDimOrScalar(params.stride, ndim, dimIdx, true);

  let shape: number;
  if (!params.autoPad || params.autoPad === 'NOTSET') {
    if (params.ceilMode) {
      // ceilModeにおいて、右端のpaddingしか含まれないウィンドウは除外する
      shape = Math.ceil((width + padding0 - kernelSize) / stride) + 1;
    } else {
      shape =
        Math.floor((width + padding0 + padding1 - kernelSize) / stride) + 1;
    }
  } else if (
    params.autoPad === 'SAME_LOWER' ||
    params.autoPad === 'SAME_UPPER'
  ) {
    // calculate output shape as if padding is zero
    shape = Math.ceil(width / stride);
    const sumPad = (shape - 1) * stride + kernelSize - width;
    if (params.autoPad === 'SAME_LOWER') {
      padding0 = Math.ceil(sumPad / 2);
      padding1 = Math.floor(sumPad / 2);
    } else {
      padding0 = Math.floor(sumPad / 2);
      padding1 = Math.ceil(sumPad / 2);
    }
  } else if (params.autoPad === 'VALID') {
    shape = Math.ceil((width - kernelSize + 1) / stride);
    padding0 = padding1 = 0;
  } else {
    throw new Error(`Unknown autoPad ${params.autoPad}`);
  }

  return { shape, kernelSize, padding0, padding1, stride };
}

export function avgPool2DCalcShape(
  params: AvgPoolParams,
  dimsX: ReadonlyArray<number>
) {
  const [batch, ch] = dimsX;
  const inShape: number[] = [];
  const kernelShape: number[] = [];
  const pads: number[] = [0, 0, 0, 0]; // [ybegin, xbegin, yend, xend]
  const strides: number[] = [];
  const outShape: number[] = [];
  for (let dim = 0; dim < 2; dim++) {
    const width = dimsX[2 + dim];
    const { shape, kernelSize, padding0, padding1, stride } =
      avgPoolCalcAxisShape(params, 2, dim, width);
    inShape.push(width);
    kernelShape.push(kernelSize);
    pads[dim] = padding0;
    pads[dim + 2] = padding1;
    strides.push(stride);
    outShape.push(shape);
  }

  return {
    batch,
    kernelShape,
    pads,
    strides,
    inShape,
    outShape,
    ch,
  };
}

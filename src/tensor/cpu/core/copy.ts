import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor } from '../cpuTensor';
import { getBroadcastStride } from '../../shapeUtil';

function broadcastCopy0(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  xStrides: ReadonlyArray<number>
): void {
  dy[0] = dx[0];
}

function broadcastCopy1(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0] = toShape;
  const [xt0] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    dy[id++] = dx[i0 * xt0];
  }
}

function broadcastCopy2(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1] = toShape;
  const [xt0, xt1] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      dy[id++] = dx[i0 * xt0 + i1 * xt1];
    }
  }
}

function broadcastCopy3(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2] = toShape;
  const [xt0, xt1, xt2] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        dy[id++] = dx[i0 * xt0 + i1 * xt1 + i2 * xt2];
      }
    }
  }
}

function broadcastCopy4(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3] = toShape;
  const [xt0, xt1, xt2, xt3] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          dy[id++] = dx[i0 * xt0 + i1 * xt1 + i2 * xt2 + i3 * xt3];
        }
      }
    }
  }
}

function broadcastCopy5(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3, ts4] = toShape;
  const [xt0, xt1, xt2, xt3, xt4] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          for (let i4 = 0; i4 < ts4; i4++) {
            dy[id++] = dx[i0 * xt0 + i1 * xt1 + i2 * xt2 + i3 * xt3 + i4 * xt4];
          }
        }
      }
    }
  }
}

function broadcastCopy6(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3, ts4, ts5] = toShape;
  const [xt0, xt1, xt2, xt3, xt4, xt5] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          for (let i4 = 0; i4 < ts4; i4++) {
            for (let i5 = 0; i5 < ts5; i5++) {
              dy[id++] =
                dx[
                  i0 * xt0 +
                    i1 * xt1 +
                    i2 * xt2 +
                    i3 * xt3 +
                    i4 * xt4 +
                    i5 * xt5
                ];
            }
          }
        }
      }
    }
  }
}

function broadcastCopy7(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3, ts4, ts5, ts6] = toShape;
  const [xt0, xt1, xt2, xt3, xt4, xt5, xt6] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          for (let i4 = 0; i4 < ts4; i4++) {
            for (let i5 = 0; i5 < ts5; i5++) {
              for (let i6 = 0; i6 < ts6; i6++) {
                dy[id++] =
                  dx[
                    i0 * xt0 +
                      i1 * xt1 +
                      i2 * xt2 +
                      i3 * xt3 +
                      i4 * xt4 +
                      i5 * xt5 +
                      i6 * xt6
                  ];
              }
            }
          }
        }
      }
    }
  }
}

function broadcastCopy8(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3, ts4, ts5, ts6, ts7] = toShape;
  const [xt0, xt1, xt2, xt3, xt4, xt5, xt6, xt7] = xStrides;
  let id = 0;
  for (let i0 = 0; i0 < ts0; i0++) {
    for (let i1 = 0; i1 < ts1; i1++) {
      for (let i2 = 0; i2 < ts2; i2++) {
        for (let i3 = 0; i3 < ts3; i3++) {
          for (let i4 = 0; i4 < ts4; i4++) {
            for (let i5 = 0; i5 < ts5; i5++) {
              for (let i6 = 0; i6 < ts6; i6++) {
                for (let i7 = 0; i7 < ts7; i7++) {
                  dy[id++] =
                    dx[
                      i0 * xt0 +
                        i1 * xt1 +
                        i2 * xt2 +
                        i3 * xt3 +
                        i4 * xt4 +
                        i5 * xt5 +
                        i6 * xt6 +
                        i7 * xt7
                    ];
                }
              }
            }
          }
        }
      }
    }
  }
}

export function broadcastCopyND(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>
) {
  switch (toShape.length) {
    case 0:
      broadcastCopy0(dx, dy, toShape, xStrides);
      break;
    case 1:
      broadcastCopy1(dx, dy, toShape, xStrides);
      break;
    case 2:
      broadcastCopy2(dx, dy, toShape, xStrides);
      break;
    case 3:
      broadcastCopy3(dx, dy, toShape, xStrides);
      break;
    case 4:
      broadcastCopy4(dx, dy, toShape, xStrides);
      break;
    case 5:
      broadcastCopy5(dx, dy, toShape, xStrides);
      break;
    case 6:
      broadcastCopy6(dx, dy, toShape, xStrides);
      break;
    case 7:
      broadcastCopy7(dx, dy, toShape, xStrides);
      break;
    case 8:
      broadcastCopy8(dx, dy, toShape, xStrides);
      break;
    default:
      throw new Error(
        `broadcastCopyND to dimension ${toShape.length} is not yet supported.`
      );
  }
}

export function broadcastTo(
  x: CPUTensor,
  shape: ReadonlyArray<number>
): CPUTensor {
  const xStride = getBroadcastStride(x.shape, shape);
  const y = CPUTensor.zeros(shape, x.dtype);
  const dx = x.getBuffer().data;
  const dy = y.getBuffer().data;
  broadcastCopyND(dx, dy, shape, xStride);

  return y;
}

export function stridedCopy(
  x: CPUTensor,
  newShape: ReadonlyArray<number>,
  xStride: ReadonlyArray<number>
): CPUTensor {
  const y = CPUTensor.zeros(newShape, x.dtype);
  const dx = x.getBuffer().data;
  const dy = y.getBuffer().data;
  broadcastCopyND(dx, dy, newShape, xStride);

  return y;
}

import { TypedArrayTypes } from '../../dtype';
import { CPUTensor } from '../cpuTensor';
import { getReductionByAxis, getReductionByBroadcastShape } from '../shapeUtil';

export function sumReduction00(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShapeStrides: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionStrides: ReadonlyArray<number>
): void {
  let id = 0;
  let sum = 0.0;
  sum += dx[0];
  dy[id++] = sum;
}

function sumReduction01(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [rs0] = reductionShape;
  const [rt0] = reductionStrides;
  let id = 0;
  let sum = 0.0;
  for (let r0 = 0; r0 < rs0; r0++) {
    sum += dx[r0 * rt0];
  }
  dy[id++] = sum;
}
function sumReduction02(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [rs0, rs1] = reductionShape;
  const [rt0, rt1] = reductionStrides;
  let id = 0;
  let sum = 0.0;
  for (let r0 = 0; r0 < rs0; r0++) {
    for (let r1 = 0; r1 < rs1; r1++) {
      sum += dx[r0 * rt0 + r1 * rt1];
    }
  }
  dy[id++] = sum;
}

function sumReduction03(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [rs0, rs1, rs2] = reductionShape;
  const [rt0, rt1, rt2] = reductionStrides;
  let id = 0;
  let sum = 0.0;
  for (let r0 = 0; r0 < rs0; r0++) {
    for (let r1 = 0; r1 < rs1; r1++) {
      for (let r2 = 0; r2 < rs2; r2++) {
        sum += dx[r0 * rt0 + r1 * rt1 + r2 * rt2];
      }
    }
  }
  dy[id++] = sum;
}

function sumReduction04(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [rs0, rs1, rs2, rs3] = reductionShape;
  const [rt0, rt1, rt2, rt3] = reductionStrides;
  let id = 0;
  let sum = 0.0;
  for (let r0 = 0; r0 < rs0; r0++) {
    for (let r1 = 0; r1 < rs1; r1++) {
      for (let r2 = 0; r2 < rs2; r2++) {
        for (let r3 = 0; r3 < rs3; r3++) {
          sum += dx[r0 * rt0 + r1 * rt1 + r2 * rt2 + r3 * rt3];
        }
      }
    }
  }
  dy[id++] = sum;
}

function sumReduction10(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0] = toShape;
  const [tt0] = toShapeStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    let sum = 0.0;
    sum += dx[t0 * tt0];
    dy[id++] = sum;
  }
}
function sumReduction11(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0] = toShape;
  const [tt0] = toShapeStrides;
  const [rs0] = reductionShape;
  const [rt0] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    let sum = 0.0;
    for (let r0 = 0; r0 < rs0; r0++) {
      sum += dx[t0 * tt0 + r0 * rt0];
    }
    dy[id++] = sum;
  }
}
function sumReduction12(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0] = toShape;
  const [tt0] = toShapeStrides;
  const [rs0, rs1] = reductionShape;
  const [rt0, rt1] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    let sum = 0.0;
    for (let r0 = 0; r0 < rs0; r0++) {
      for (let r1 = 0; r1 < rs1; r1++) {
        sum += dx[t0 * tt0 + r0 * rt0 + r1 * rt1];
      }
    }
    dy[id++] = sum;
  }
}

function sumReduction13(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0] = toShape;
  const [tt0] = toShapeStrides;
  const [rs0, rs1, rs2] = reductionShape;
  const [rt0, rt1, rt2] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    let sum = 0.0;
    for (let r0 = 0; r0 < rs0; r0++) {
      for (let r1 = 0; r1 < rs1; r1++) {
        for (let r2 = 0; r2 < rs2; r2++) {
          sum += dx[t0 * tt0 + r0 * rt0 + r1 * rt1 + r2 * rt2];
        }
      }
    }
    dy[id++] = sum;
  }
}

function sumReduction20(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1] = toShape;
  const [tt0, tt1] = toShapeStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      let sum = 0.0;
      sum += dx[t0 * tt0 + t1 * tt1];
      dy[id++] = sum;
    }
  }
}

function sumReduction21(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1] = toShape;
  const [tt0, tt1] = toShapeStrides;
  const [rs0] = reductionShape;
  const [rt0] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      let sum = 0.0;
      for (let r0 = 0; r0 < rs0; r0++) {
        sum += dx[t0 * tt0 + t1 * tt1 + r0 * rt0];
      }
      dy[id++] = sum;
    }
  }
}

function sumReduction22(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1] = toShape;
  const [tt0, tt1] = toShapeStrides;
  const [rs0, rs1] = reductionShape;
  const [rt0, rt1] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      let sum = 0.0;
      for (let r0 = 0; r0 < rs0; r0++) {
        for (let r1 = 0; r1 < rs1; r1++) {
          sum += dx[t0 * tt0 + t1 * tt1 + r0 * rt0 + r1 * rt1];
        }
      }
      dy[id++] = sum;
    }
  }
}

function sumReduction30(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2] = toShape;
  const [tt0, tt1, tt2] = toShapeStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      for (let t2 = 0; t2 < ts2; t2++) {
        let sum = 0.0;
        sum += dx[t0 * tt0 + t1 * tt1 + t2 * tt2];
        dy[id++] = sum;
      }
    }
  }
}

function sumReduction31(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2] = toShape;
  const [tt0, tt1, tt2] = toShapeStrides;
  const [rs0] = reductionShape;
  const [rt0] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      for (let t2 = 0; t2 < ts2; t2++) {
        let sum = 0.0;
        for (let r0 = 0; r0 < rs0; r0++) {
          sum += dx[t0 * tt0 + t1 * tt1 + t2 * tt2 + r0 * rt0];
        }
        dy[id++] = sum;
      }
    }
  }
}

function sumReduction33(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2] = toShape;
  const [tt0, tt1, tt2] = toShapeStrides;
  const [rs0, rs1, rs2] = reductionShape;
  const [rt0, rt1, rt2] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      for (let t2 = 0; t2 < ts2; t2++) {
        let sum = 0.0;
        for (let r0 = 0; r0 < rs0; r0++) {
          for (let r1 = 0; r1 < rs1; r1++) {
            for (let r2 = 0; r2 < rs2; r2++) {
              sum +=
                dx[
                  t0 * tt0 +
                    t1 * tt1 +
                    t2 * tt2 +
                    r0 * rt0 +
                    r1 * rt1 +
                    r2 * rt2
                ];
            }
          }
        }
        dy[id++] = sum;
      }
    }
  }
}

function sumReduction44(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: ReadonlyArray<number>,
  toShapeStrides: ReadonlyArray<number>,
  reductionShape: ReadonlyArray<number>,
  reductionStrides: ReadonlyArray<number>
): void {
  const [ts0, ts1, ts2, ts3] = toShape;
  const [tt0, tt1, tt2, tt3] = toShapeStrides;
  const [rs0, rs1, rs2, rs3] = reductionShape;
  const [rt0, rt1, rt2, rt3] = reductionStrides;
  let id = 0;
  for (let t0 = 0; t0 < ts0; t0++) {
    for (let t1 = 0; t1 < ts1; t1++) {
      for (let t2 = 0; t2 < ts2; t2++) {
        for (let t3 = 0; t3 < ts3; t3++) {
          let sum = 0.0;
          for (let r0 = 0; r0 < rs0; r0++) {
            for (let r1 = 0; r1 < rs1; r1++) {
              for (let r2 = 0; r2 < rs2; r2++) {
                for (let r3 = 0; r3 < rs3; r3++) {
                  sum +=
                    dx[
                      t0 * tt0 +
                        t1 * tt1 +
                        t2 * tt2 +
                        t3 * tt3 +
                        r0 * rt0 +
                        r1 * rt1 +
                        r2 * rt2 +
                        r3 * rt3
                    ];
                }
              }
            }
          }
          dy[id++] = sum;
        }
      }
    }
  }
}

function dispatchSum(
  dx: TypedArrayTypes,
  dy: TypedArrayTypes,
  toShape: number[],
  toShapeStrides: number[],
  reductionShape: number[],
  reductionStrides: number[]
): void {
  switch (toShape.length) {
    case 0:
      switch (reductionShape.length) {
        case 0:
          sumReduction00(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 1:
          sumReduction01(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 2:
          sumReduction02(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 3:
          sumReduction03(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 4:
          sumReduction04(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        default:
          throw new Error('sum not implemented');
      }
      break;
    case 1:
      switch (reductionShape.length) {
        case 0:
          sumReduction10(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 1:
          sumReduction11(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 2:
          sumReduction12(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 3:
          sumReduction13(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        default:
          throw new Error('sum not implemented');
      }
      break;
    case 2:
      switch (reductionShape.length) {
        case 0:
          sumReduction20(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 1:
          sumReduction21(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 2:
          sumReduction22(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        default:
          throw new Error('sum not implemented');
      }
      break;
    case 3:
      switch (reductionShape.length) {
        case 0:
          sumReduction30(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 1:
          sumReduction31(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        case 3:
          sumReduction33(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        default:
          throw new Error('sum not implemented');
      }
      break;
    case 4:
      switch (reductionShape.length) {
        case 4:
          sumReduction44(
            dx,
            dy,
            toShape,
            toShapeStrides,
            reductionShape,
            reductionStrides
          );
          break;
        default:
          throw new Error('sum not implemented');
      }
      break;
    default:
      throw new Error('sum not implemented');
  }
}

export function sum(
  x: CPUTensor,
  axis?: number | number[] | null,
  keepdims?: boolean
): CPUTensor {
  const {
    toShape,
    toShapeStrides,
    toShapeKeepdims,
    reductionShape,
    reductionStrides,
  } = getReductionByAxis(x.shape, axis);
  const y = CPUTensor.zeros(keepdims ? toShapeKeepdims : toShape, x.dtype);
  const dx = x.getBuffer().data;
  const dy = y.getBuffer().data;
  dispatchSum(
    dx,
    dy,
    toShape,
    toShapeStrides,
    reductionShape,
    reductionStrides
  );

  return y;
}

export function sumTo(x: CPUTensor, shape: ReadonlyArray<number>): CPUTensor {
  const {
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides,
  } = getReductionByBroadcastShape(x.shape, shape);
  const y = CPUTensor.zeros(shape, x.dtype);
  const dx = x.getBuffer().data;
  const dy = y.getBuffer().data;
  dispatchSum(
    dx,
    dy,
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides
  );

  return y;
}

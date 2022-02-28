/* ブロードキャスト等の形状操作ユーティリティ
 */

import { arange, arrayProd } from '../util';

export function getStride(shape: ReadonlyArray<number>): number[] {
  const strides: number[] = [];
  let size = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(size);
    size *= shape[i];
  }
  return strides;
}

export function getBroadcastStride(
  fromShape: ReadonlyArray<number>,
  toShape: ReadonlyArray<number>
): number[] {
  // fromShapeがtoShapeより長い場合、fromShapeの先頭の余分な要素がすべて1ならOKで、先頭要素を削除
  // fromShapeがtoShapeより短ければ、先頭に長さ1の次元を付与
  // fromShapeの各次元がtoShapeと一致するか確認。長さ1ならそこはbroadcast対象でstride=0となる。
  // toShapeのインデックスと内積して、fromShapeに対応するテンソルの要素のインデックスが得られるstrideを返す
  const expandedFromShape = [...fromShape];
  while (expandedFromShape.length > toShape.length) {
    const p = expandedFromShape.shift();
    if (p !== 1) {
      throw new Error('fromShape is longer than toShape and length is not 1');
    }
  }
  while (expandedFromShape.length < toShape.length) {
    expandedFromShape.unshift(1);
  }
  const strides: number[] = [];
  let size = 1;
  for (let dim = toShape.length - 1; dim >= 0; dim--) {
    const f = expandedFromShape[dim];
    const t = toShape[dim];
    if (f === t) {
      strides.unshift(size);
    } else if (f === 1) {
      strides.unshift(0); // broadcastされる軸
    } else {
      throw new Error(`axis length does not match: ${fromShape} vs ${toShape}`);
    }
    size *= f;
  }
  return strides;
}

export function getMultiBroadcastShape(
  shapes: ReadonlyArray<ReadonlyArray<number>>
): { shape: number[]; allStrides: number[][] } {
  const resultDim = Math.max(...shapes.map((s) => s.length));
  // 先頭に長さ1の次元をつけて、次元数をそろえる
  const expandedShapes: number[][] = [];
  for (const shape of shapes) {
    const expandedShape: number[] = [...shape];
    while (expandedShape.length < resultDim) {
      expandedShape.unshift(1);
    }
    expandedShapes.push(expandedShape);
  }
  // 長さが1でないものに合わせる
  const broadcastedShape: number[] = [];
  for (let dim = 0; dim < resultDim; dim++) {
    let axisLength = 1;
    for (const shape of expandedShapes) {
      if (axisLength === 1) {
        if (axisLength !== shape[dim]) {
          axisLength = shape[dim];
        }
      } else if (shape[dim] !== 1 && axisLength !== shape[dim]) {
        throw new Error(`axis length does not match: ${shapes}`);
      }
    }
    broadcastedShape.push(axisLength);
  }
  // 各テンソルを読みだすときのstrideを計算
  const allStrides = shapes.map((s) => getBroadcastStride(s, broadcastedShape));
  return { shape: broadcastedShape, allStrides };
}

export function getReductionByAxis(
  fromShape: ReadonlyArray<number>,
  axis?: number | number[] | null
): {
  toShape: number[];
  toShapeStrides: number[];
  toShapeKeepdims: number[];
  reductionShape: number[];
  reductionStrides: number[];
} {
  let axes: number[];
  if (axis == null) {
    axes = arange(fromShape.length);
  } else if (typeof axis === 'number') {
    axes = [axis];
  } else if (Array.isArray(axis)) {
    axes = axis;
  } else {
    throw new Error(`axis is not number nor Array.`);
  }
  let fromSize = 1;
  const toShape: number[] = [];
  const toShapeKeepdims: number[] = [];
  const toShapeStrides: number[] = [];
  const reductionShape: number[] = [];
  const reductionStrides: number[] = [];
  for (let dim = fromShape.length - 1; dim >= 0; dim--) {
    const f = fromShape[dim];
    if (axes.includes(dim)) {
      toShapeKeepdims.unshift(1);
      reductionShape.unshift(f);
      reductionStrides.unshift(fromSize);
    } else {
      toShapeKeepdims.unshift(f);
      toShape.unshift(f);
      toShapeStrides.unshift(fromSize);
    }

    fromSize *= f;
  }

  return {
    toShape,
    toShapeKeepdims,
    toShapeStrides,
    reductionShape,
    reductionStrides,
  };
}

export function getReductionByBroadcastShape(
  fromShape: ReadonlyArray<number>,
  toShape: ReadonlyArray<number>
): {
  toShapeSqueeze: number[];
  toShapeSqueezeFromStrides: number[];
  reductionShape: number[];
  reductionStrides: number[];
} {
  // toShapeからfromShapeへbroadcastする場合の逆変換でreductionする場合のインデックス計算
  const expandedToShape: number[] = [...toShape];
  while (expandedToShape.length < fromShape.length) {
    expandedToShape.unshift(1);
  }

  let fromSize = 1;
  const toShapeSqueeze: number[] = [];
  const toShapeSqueezeFromStrides: number[] = [];
  const reductionShape: number[] = [];
  const reductionStrides: number[] = [];
  for (let dim = fromShape.length - 1; dim >= 0; dim--) {
    const t = expandedToShape[dim];
    const f = fromShape[dim];
    if (t === 1 && f !== t) {
      // broadcast軸
      reductionShape.unshift(f);
      reductionStrides.unshift(fromSize);
    } else {
      toShapeSqueeze.unshift(t);
      toShapeSqueezeFromStrides.unshift(fromSize);
    }

    fromSize *= f;
  }

  return {
    toShapeSqueeze,
    toShapeSqueezeFromStrides,
    reductionShape,
    reductionStrides,
  };
}

export function calcReshape(
  xShape: ReadonlyArray<number>,
  shape: ReadonlyArray<number> | number,
  allowZero: boolean
): number[] {
  const xSize = arrayProd(xShape);
  let shapeArray: ReadonlyArray<number>;
  if (typeof shape === 'number') {
    shapeArray = [shape];
  } else {
    shapeArray = shape;
  }
  let nonMinusProd = 1;
  let minusAxis: number | null = null;
  const newShape: number[] = [];
  for (let dim = 0; dim < shapeArray.length; dim++) {
    let s = shapeArray[dim];
    if (s < 0) {
      if (minusAxis !== null) {
        throw new Error('Multiple -1 value in shape');
      }
      minusAxis = dim;
    } else {
      if (s === 0) {
        if (!allowZero) {
          // ONNXのReshapeオペレータの機能
          // copy original value from x.shape
          if (xShape.length < dim) {
            throw new Error('No corresponding input shape axis for zero');
          }
          s = xShape[dim];
        }
      }
      nonMinusProd *= s;
    }
    newShape.push(s);
  }
  if (minusAxis !== null) {
    if (nonMinusProd === 0) {
      throw new Error('Cannot determine size for -1: zero division');
    }
    if (xSize % nonMinusProd !== 0) {
      throw new Error('Cannot determine size for -1: non-integer result');
    }
    const minusAxisSize = xSize / nonMinusProd;
    newShape[minusAxis] = minusAxisSize;
  } else {
    if (nonMinusProd !== xSize) {
      throw new Error('Size does not match');
    }
  }
  return newShape;
}

export function calcTransposeShape(
  xShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  axes?: ReadonlyArray<number> | null
): { newShape: number[]; srcStrides: number[] } {
  let axesChecked: ReadonlyArray<number>;
  if (axes) {
    if (axes.length !== xShape.length) {
      throw new Error('length of axes does not match with x');
    }
    // ほか、すべての軸が1つずつ使われているかのチェックをすべき
    axesChecked = axes;
  } else {
    axesChecked = arange(xShape.length - 1, -1, -1); // 逆順[x.ndim-1, x.ndim-2, ..., 2, 1, 0]
  }
  const newShape: number[] = [];
  const srcStrides: number[] = [];
  for (let dim = 0; dim < axesChecked.length; dim++) {
    const ax = axesChecked[dim];
    newShape.push(xShape[ax]);
    srcStrides.push(xStrides[ax]);
  }
  return { newShape, srcStrides };
}

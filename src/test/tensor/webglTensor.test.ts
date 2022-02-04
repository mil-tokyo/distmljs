import { assert } from 'chai';
import {
  getTensorTextureShapeFormatForDType,
  tensorTextureShapeFormatRGBA16F,
  WebGLTensor,
} from '../../tensor/webgl/webglTensor';
import { arange } from '../../util';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

describe('webglTensor', () => {
  if (!testFlag.webgl) {
    return;
  }

  describe('basic', () => {
    it('computes size', () => {
      const t = WebGLTensor.zeros([3, 4]);
      assert.equal(t.size, 12);
      assert.equal(t.ndim, 2);
      assert.deepEqual(t.shape, [3, 4]);
      assert.deepEqual(t.strides, [4, 1]);
    });

    it('computes size of scalar', () => {
      const t = WebGLTensor.zeros([]);
      assert.equal(t.size, 1);
      assert.equal(t.ndim, 0);
      assert.deepEqual(t.shape, []);
      assert.deepEqual(t.strides, []);
    });

    it('create from array', async () => {
      const t = WebGLTensor.fromArray([10, 20, 30, 4.5, 50, 60], [2, 3]);
      const cpu = await t.to('cpu');
      assert.equal(cpu.get(1, 0), 4.5); // float32 / float16で厳密に一致する
    });

    it('create from array int32', async () => {
      // float32では正確に表せない数値
      const t = WebGLTensor.fromArray(
        [10, 20, 30, 16843009, 16843010, 16843011],
        [2, 3],
        'int32'
      );
      const cpu = await t.to('cpu');
      assert.equal(cpu.get(1, 0), 16843009);
      assert.equal(cpu.get(1, 1), 16843010);
      assert.equal(cpu.get(1, 2), 16843011);
    });

    it('create from array uint8', async () => {
      const t = WebGLTensor.fromArray(
        [10, 20, 30, 40, 50, 60],
        [2, 3],
        'uint8'
      );
      const cpu = await t.to('cpu');
      assert.equal(cpu.get(1, 0), 40);
    });

    it('create from array bool', async () => {
      const t = WebGLTensor.fromArray([0, 1, 0, 1, 1, 0], [2, 3], 'bool');
      const cpu = await t.to('cpu');
      assert.equal(cpu.get(1, 0), 1);
    });

    it('create from array 2DArray', async () => {
      const t = WebGLTensor.empty([2, 3], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 2,
        height: 2,
        depth: 2,
      });
      t.setArray([10, 20, 30, 4.5, 50, 60]);
      assert.deepEqual(await t.toArrayAsync(), [10, 20, 30, 4.5, 50, 60]);
    });
  });

  describe('exp', () => {
    it('exp', async () => {
      const x = WebGLTensor.fromArray([1, -1], [2]);
      const y = WebGLTensor.exp(x);
      arrayNearlyEqual(await y.toArrayAsync(), [2.71828182845904, 0.367879441]);
    });

    it('exp 2darray', async () => {
      const x = WebGLTensor.empty([2], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 1,
        height: 1,
        depth: 2,
      });
      x.setArray([1, -1]);
      const y = WebGLTensor.exp(x);
      arrayNearlyEqual(await y.toArrayAsync(), [2.71828182845904, 0.367879441]);
    });
  });

  describe('abs', () => {
    it('abs', async () => {
      const x = WebGLTensor.fromArray([1.5, -3.5], [2]);
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [1.5, 3.5]);
    });

    it('abs 2darray', async () => {
      const x = WebGLTensor.empty([2], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 1,
        height: 1,
        depth: 2,
      });
      x.setArray([1.5, -3.5]);
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [1.5, 3.5]);
    });

    it('abs int32', async () => {
      const x = WebGLTensor.fromArray([16843010, -16843009], [2], 'int32');
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [16843010, 16843009]);
    });
  });

  describe('add', () => {
    it('add', async () => {
      const lhs = WebGLTensor.fromArray([10, 20], [2]);
      const rhs = WebGLTensor.fromArray([50, 60], [2]);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [60, 80]);
    });

    it('add int32', async () => {
      const lhs = WebGLTensor.fromArray([16843009, 16843010], [2], 'int32');
      const rhs = WebGLTensor.fromArray([1, -3], [2], 'int32');
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [16843010, 16843007]);
    });

    it('add uint8', async () => {
      const lhs = WebGLTensor.fromArray([100, 101], [2], 'uint8');
      const rhs = WebGLTensor.fromArray([1, 3], [2], 'uint8');
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [101, 104]);
    });

    it('broadcast 0d to 1d', async () => {
      const lhs = WebGLTensor.fromArray([10, 20], [2]);
      const rhs = WebGLTensor.fromArray([100], []);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [110, 120]);
    });

    it('broadcast 1d[1,2] to 2d', async () => {
      const lhs = WebGLTensor.fromArray([10, 20, 30, 40], [2, 2]);
      const rhs = WebGLTensor.fromArray([100, 200], [1, 2]);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [110, 220, 130, 240]);
    });

    it('broadcast 1d[2,1] to 2d', async () => {
      const lhs = WebGLTensor.fromArray([10, 20, 30, 40], [2, 2]);
      const rhs = WebGLTensor.fromArray([100, 200], [2, 1]);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [110, 120, 230, 240]);
    });

    it('broadcast 1d[2,1] to 2d 2darray', async () => {
      const lhs = WebGLTensor.empty([2, 2], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 1,
        height: 2,
        depth: 2,
      });
      lhs.setArray([10, 20, 30, 40]);
      const rhs = WebGLTensor.fromArray([100, 200], [2, 1]);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [110, 120, 230, 240]);
      // TODO: test when output is 2DArray (mock WebGLTensor.empty is needed)
    });
  });

  // describe('mul', () => {
  //   it('mul', async () => {
  //     const lhs = WebGLTensor.fromArray([10, 20], [2]);
  //     const rhs = WebGLTensor.fromArray([50, 60], [2]);
  //     const y = WebGLTensor.mul(lhs, rhs);
  //     assert.deepEqual(y.toArray(), [500, 1200]);
  //   });
  // });

  // describe('dot', () => {
  //   it('dot', () => {
  //     const lhs = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const rhs = WebGLTensor.fromArray(
  //       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  //       [3, 4]
  //     );
  //     const y = WebGLTensor.gemm(lhs, rhs);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [50, 53, 56, 59, 176, 188, 200, 212]);
  //   });
  // });

  // describe('gemm', () => {
  //   it('gemm_n_n', () => {
  //     const lhs = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const rhs = WebGLTensor.fromArray(
  //       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  //       [3, 4]
  //     );
  //     const y = WebGLTensor.gemm(lhs, rhs);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [50, 53, 56, 59, 176, 188, 200, 212]);
  //   });
  //   it('gemm_n_t', () => {
  //     const lhs = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const rhs = WebGLTensor.fromArray(
  //       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  //       [4, 3]
  //     );
  //     const y = WebGLTensor.gemm(lhs, rhs, false, true);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [35, 44, 53, 62, 134, 170, 206, 242]);
  //   });
  //   it('gemm_t_n', () => {
  //     const lhs = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [3, 2]);
  //     const rhs = WebGLTensor.fromArray(
  //       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  //       [3, 4]
  //     );
  //     const y = WebGLTensor.gemm(lhs, rhs, true, false);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [100, 106, 112, 118, 142, 151, 160, 169]);
  //   });
  //   it('gemm_t_t', () => {
  //     const lhs = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [3, 2]);
  //     const rhs = WebGLTensor.fromArray(
  //       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  //       [4, 3]
  //     );
  //     const y = WebGLTensor.gemm(lhs, rhs, true, true);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [70, 88, 106, 124, 103, 130, 157, 184]);
  //   });
  // });

  // describe('broadcast', () => {
  //   it('2d to 4d', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.broadcastTo(x, [2, 4, 2, 3]);
  //     assert.deepEqual(y.shape, [2, 4, 2, 3]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
  //         0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
  //       ]
  //     );
  //   });
  //   it('3d to 4d', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 1, 3]);
  //     const y = WebGLTensor.broadcastTo(x, [4, 2, 2, 3]);
  //     assert.deepEqual(y.shape, [4, 2, 2, 3]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
  //         0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
  //       ]
  //     );
  //   });
  //   it('0d to 0d', () => {
  //     const x = WebGLTensor.fromArray([3], []);
  //     const y = WebGLTensor.broadcastTo(x, []);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [3]);
  //   });
  //   it('0d to 2d', () => {
  //     const x = WebGLTensor.fromArray([3], []);
  //     const y = WebGLTensor.broadcastTo(x, [2, 3]);
  //     assert.deepEqual(y.shape, [2, 3]);
  //     assert.deepEqual(y.toArray(), [3, 3, 3, 3, 3, 3]);
  //   });
  //   // TODO: add cases
  // });

  // describe('broadcastShapes', () => {
  //   it('2d', () => {
  //     assert.deepEqual(
  //       WebGLTensor.broadcastShapes([
  //         [1, 3],
  //         [2, 1],
  //       ]),
  //       [2, 3]
  //     );
  //     assert.deepEqual(WebGLTensor.broadcastShapes([[3], [2, 1]]), [2, 3]);
  //     assert.deepEqual(WebGLTensor.broadcastShapes([[3], [2, 3]]), [2, 3]);
  //     assert.deepEqual(WebGLTensor.broadcastShapes([[], [2, 3]]), [2, 3]);
  //   });
  //   // TODO: add cases
  // });

  // describe('sum', () => {
  //   it('2dto1d axis0', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, 0);
  //     assert.deepEqual(y.shape, [3]);
  //     assert.deepEqual(y.toArray(), [3, 5, 7]);
  //   });
  //   it('2dto1d axis0 keepdims', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, 0, true);
  //     assert.deepEqual(y.shape, [1, 3]);
  //     assert.deepEqual(y.toArray(), [3, 5, 7]);
  //   });
  //   it('2dto1d axis1', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, 1);
  //     assert.deepEqual(y.shape, [2]);
  //     assert.deepEqual(y.toArray(), [3, 12]);
  //   });
  //   it('2dto1d axis1 keepdims', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, 1, true);
  //     assert.deepEqual(y.shape, [2, 1]);
  //     assert.deepEqual(y.toArray(), [3, 12]);
  //   });
  //   it('2dto0d axis[0,1]', () => {
  //     // scalar
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, [0, 1]);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [15]);
  //   });
  //   it('2dto0d axisnull', () => {
  //     // scalar
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [15]);
  //   });
  //   it('2dto0d axisnull keepdims', () => {
  //     // scalar
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sum(x, null, true);
  //     assert.deepEqual(y.shape, [1, 1]);
  //     assert.deepEqual(y.toArray(), [15]);
  //   });
  //   it('3dto2d axis0', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sum(x, 0);
  //     assert.deepEqual(y.shape, [3, 4]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
  //     );
  //   });
  //   it('3dto2d axis1', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sum(x, 1);
  //     assert.deepEqual(y.shape, [2, 4]);
  //     assert.deepEqual(y.toArray(), [12, 15, 18, 21, 48, 51, 54, 57]);
  //   });
  //   it('3dto2d axis2', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sum(x, 2);
  //     assert.deepEqual(y.shape, [2, 3]);
  //     assert.deepEqual(y.toArray(), [6, 22, 38, 54, 70, 86]);
  //   });
  //   it('3dto0d axisnull', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sum(x);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [276]);
  //   });
  //   it('3dto1d axis[0,2]', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sum(x, [0, 2]);
  //     assert.deepEqual(y.shape, [3]);
  //     assert.deepEqual(y.toArray(), [60, 92, 124]);
  //   });
  //   it('4dto2d axis[0,2]', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
  //     const y = WebGLTensor.sum(x, [0, 2]);
  //     assert.deepEqual(y.shape, [3, 5]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         300, 308, 316, 324, 332, 460, 468, 476, 484, 492, 620, 628, 636, 644,
  //         652,
  //       ]
  //     );
  //   });
  //   it('4dto3d axis1 keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
  //     const y = WebGLTensor.sum(x, 1, true);
  //     assert.deepEqual(y.shape, [2, 1, 4, 5]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108,
  //         111, 114, 117, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270,
  //         273, 276, 279, 282, 285, 288, 291, 294, 297,
  //       ]
  //     );
  //   });
  // });

  // describe('sumTo', () => {
  //   it('2dto1d axis0', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sumTo(x, [3]);
  //     assert.deepEqual(y.shape, [3]);
  //     assert.deepEqual(y.toArray(), [3, 5, 7]);
  //   });
  //   it('2dto1d axis0 keepdims', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sumTo(x, [1, 3]);
  //     assert.deepEqual(y.shape, [1, 3]);
  //     assert.deepEqual(y.toArray(), [3, 5, 7]);
  //   });
  //   it('2dto1d axis1 keepdims', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sumTo(x, [2, 1]);
  //     assert.deepEqual(y.shape, [2, 1]);
  //     assert.deepEqual(y.toArray(), [3, 12]);
  //   });
  //   it('2dto0d axisnull', () => {
  //     // scalar
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sumTo(x, []);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [15]);
  //   });
  //   it('2dto0d axisnull keepdims', () => {
  //     // scalar
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.sumTo(x, [1, 1]);
  //     assert.deepEqual(y.shape, [1, 1]);
  //     assert.deepEqual(y.toArray(), [15]);
  //   });
  //   it('3dto2d axis0', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sumTo(x, [3, 4]);
  //     assert.deepEqual(y.shape, [3, 4]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
  //     );
  //   });
  //   it('3dto2d axis1 keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sumTo(x, [2, 1, 4]);
  //     assert.deepEqual(y.shape, [2, 1, 4]);
  //     assert.deepEqual(y.toArray(), [12, 15, 18, 21, 48, 51, 54, 57]);
  //   });
  //   it('3dto2d axis2 keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sumTo(x, [2, 3, 1]);
  //     assert.deepEqual(y.shape, [2, 3, 1]);
  //     assert.deepEqual(y.toArray(), [6, 22, 38, 54, 70, 86]);
  //   });
  //   it('3dto0d axisnull', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sumTo(x, []);
  //     assert.deepEqual(y.shape, []);
  //     assert.deepEqual(y.toArray(), [276]);
  //   });
  //   it('3dto1d axis[0,2] keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.sumTo(x, [1, 3, 1]);
  //     assert.deepEqual(y.shape, [1, 3, 1]);
  //     assert.deepEqual(y.toArray(), [60, 92, 124]);
  //   });
  //   it('4dto2d axis[0,2] keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
  //     const y = WebGLTensor.sumTo(x, [1, 3, 1, 5]);
  //     assert.deepEqual(y.shape, [1, 3, 1, 5]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         300, 308, 316, 324, 332, 460, 468, 476, 484, 492, 620, 628, 636, 644,
  //         652,
  //       ]
  //     );
  //   });
  //   it('4dto3d axis1 keepdims', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
  //     const y = WebGLTensor.sumTo(x, [2, 1, 4, 5]);
  //     assert.deepEqual(y.shape, [2, 1, 4, 5]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108,
  //         111, 114, 117, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270,
  //         273, 276, 279, 282, 285, 288, 291, 294, 297,
  //       ]
  //     );
  //   });
  // });

  // describe('transpose', () => {
  //   it('2d', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.transpose(x);
  //     assert.deepEqual(y.shape, [3, 2]);
  //     assert.deepEqual(y.toArray(), [0, 3, 1, 4, 2, 5]);
  //   });
  //   it('1d', () => {
  //     // do nothing
  //     const x = WebGLTensor.fromArray([0, 1], [2]);
  //     const y = WebGLTensor.transpose(x);
  //     assert.deepEqual(y.shape, [2]);
  //     assert.deepEqual(y.toArray(), [0, 1]);
  //   });
  //   it('3d', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.transpose(x);
  //     assert.deepEqual(y.shape, [4, 3, 2]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15,
  //         7, 19, 11, 23,
  //       ]
  //     );
  //   });
  //   it('3d 2,0,1', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.transpose(x, [2, 0, 1]);
  //     assert.deepEqual(y.shape, [4, 2, 3]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7,
  //         11, 15, 19, 23,
  //       ]
  //     );
  //   });
  //   it('3d 0,2,1', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.transpose(x, [0, 2, 1]);
  //     assert.deepEqual(y.shape, [2, 4, 3]);
  //     assert.deepEqual(
  //       y.toArray(),
  //       [
  //         0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18,
  //         22, 15, 19, 23,
  //       ]
  //     );
  //   });
  // });

  // describe('reshape', () => {
  //   it('2d-2d', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.reshape(x, [1, 6]);
  //     assert.deepEqual(y.shape, [1, 6]);
  //     assert.deepEqual(y.toArray(), [0, 1, 2, 3, 4, 5]);
  //   });
  //   it('0d-2d', () => {
  //     const x = WebGLTensor.fromArray([3], []);
  //     const y = WebGLTensor.reshape(x, [1, 1]);
  //     assert.deepEqual(y.shape, [1, 1]);
  //     assert.deepEqual(y.toArray(), [3]);
  //   });
  //   it('2d-1d', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.reshape(x, 6);
  //     assert.deepEqual(y.shape, [6]);
  //   });
  //   it('2d-1d -1', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.reshape(x, -1);
  //     assert.deepEqual(y.shape, [6]);
  //   });
  //   it('2d-2d -1', () => {
  //     const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
  //     const y = WebGLTensor.reshape(x, [3, -1]);
  //     assert.deepEqual(y.shape, [3, 2]);
  //   });
  //   it('3d-2d -1', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.reshape(x, [4, -1]);
  //     assert.deepEqual(y.shape, [4, 6]);
  //   });
  //   it('3d-3d 0', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.reshape(x, [0, 2, 6], false);
  //     // allowZero=falseで0を指定すると、元のshapeの対応次元からサイズをコピー
  //     assert.deepEqual(y.shape, [2, 2, 6]);
  //   });
  //   it('3d-3d 0,-1', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     const y = WebGLTensor.reshape(x, [0, 2, -1], false);
  //     assert.deepEqual(y.shape, [2, 2, 6]);
  //   });
  //   it('3d-3d size0', () => {
  //     const x = WebGLTensor.fromArray([], [2, 0, 4]);
  //     // allowZero=true (default)だと0指定は長さ0として扱う
  //     const y = WebGLTensor.reshape(x, [0, 1]);
  //     assert.deepEqual(y.shape, [0, 1]);
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [4]);
  //     });
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [2, 3, 5]);
  //     });
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [5, -1]); //割り切れない
  //     });
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [-1, -1]); //マイナス複数
  //     });
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray([], [0, 2, 3]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [0, -1]); //サイズ0に対してマイナスは不定となるためダメ
  //     });
  //   });
  //   it('3d-3d error', () => {
  //     const x = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
  //     assert.throw(() => {
  //       WebGLTensor.reshape(x, [0, 3, 4]);
  //     });
  //   });
  // });
});

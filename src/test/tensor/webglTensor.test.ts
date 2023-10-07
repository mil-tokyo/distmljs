import { assert } from 'chai';
import {
  getTensorTextureShapeFormatForDType,
  WebGLTensor,
} from '../../tensor/webgl/webglTensor';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';
import { arange } from '../../util';

describe('webglTensor', () => {
  if (!testFlag.webgl) {
    return;
  }

  describe('basic', () => {
    it('computes size', async () => {
      const t = WebGLTensor.zeros([3, 4]);
      assert.equal(t.size, 12);
      assert.equal(t.ndim, 2);
      assert.deepEqual(t.shape, [3, 4]);
      assert.deepEqual(t.strides, [4, 1]);
    });

    it('computes size of scalar', async () => {
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
      const x = WebGLTensor.empty([2, 2, 2], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 2,
        height: 2,
        depth: 2,
      });
      x.setArray([1, -1, 2, -2, 3, -3, 4, -4]);
      const y = WebGLTensor.exp(x);
      arrayNearlyEqual(
        await y.toArrayAsync(),
        [
          2.7182817459106445, 0.3678794503211975, 7.389056205749512,
          0.1353352814912796, 20.08553695678711, 0.049787066876888275,
          54.598148345947266, 0.018315639346837997,
        ]
      );
    });
  });

  describe('abs', () => {
    it('abs', async () => {
      const x = WebGLTensor.fromArray([1.5, -3.5], [2]);
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [1.5, 3.5]);
    });

    it('abs 2darray', async () => {
      const x = WebGLTensor.empty([6], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 2,
        height: 2,
        depth: 2,
      });
      x.setArray([1.5, -3.5, 0.0, 10.25, -2.5, -1.0]);
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(
        await y.toArrayAsync(),
        [1.5, 3.5, 0.0, 10.25, 2.5, 1.0]
      );
    });

    it('abs int32', async () => {
      const x = WebGLTensor.fromArray([16843010, -16843009], [2], 'int32');
      const y = WebGLTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [16843010, 16843009]);
    });
  });

  describe('add', () => {
    it('broadcast 1d[4,1] to 2d 2darray', async () => {
      const lhs = WebGLTensor.empty([4, 2], 'float32', undefined, {
        ...getTensorTextureShapeFormatForDType('float32'),
        dim: '2DArray',
        width: 2,
        height: 2,
        depth: 2,
      });
      lhs.setArray([10, 20, 30, 40, 50, 60, 70, 80]);
      const rhs = WebGLTensor.fromArray([100, 200, 300, 400], [4, 1]);
      const y = WebGLTensor.add(lhs, rhs);
      assert.deepEqual(
        await y.toArrayAsync(),
        [110, 120, 230, 240, 350, 360, 470, 480]
      );
      // TODO: test when output is 2DArray (mock WebGLTensor.empty is needed)
    });
  });

  describe('ravel', () => {
    it('from 2d', async () => {
      const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGLTensor.ravel(x);
      assert.isTrue(x.buffer.texture === y.buffer.texture);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('flatten', () => {
    it('from 2d', async () => {
      const x = WebGLTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGLTensor.flatten(x);
      assert.isFalse(x.buffer.texture === y.buffer.texture);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });


  describe('cat', () => {
    it('cat 1d', async () => {
      const x1 = WebGLTensor.fromArray([1, 2, 3, 4], [4]);
      const x2 = WebGLTensor.fromArray([5, 6, 7, 8], [4]);
      const y = WebGLTensor.cat([x1, x2]);
      assert.deepEqual(y.shape, [8]);
      assert.deepEqual(await y.toArrayAsync(), [1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('cat 2d 1', async () => {
      const x1 = WebGLTensor.fromArray([1, 2, 3, 4], [2, 2]);
      const x2 = WebGLTensor.fromArray([5, 6, 7, 8], [2, 2]);
      const y = WebGLTensor.cat([x1, x2]);
      assert.deepEqual(y.shape, [4, 2]);
      assert.deepEqual(await y.toArrayAsync(), [1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('cat 2d 2', async () => {
      const x1 = WebGLTensor.fromArray([1, 2, 3, 4], [2, 2]);
      const x2 = WebGLTensor.fromArray([5, 6, 7, 8, 9, 10], [2, 3]);
      const y = WebGLTensor.cat([x1, x2], 1);
      assert.deepEqual(y.shape, [2, 5]);
      assert.deepEqual(await y.toArrayAsync(), [1, 2, 5, 6, 7, 3, 4, 8, 9, 10]);
    });

    it('cat 2d 3', async () => {
      const x1 = WebGLTensor.fromArray(arange(2 * 3), [2, 3]);
      const x2 = WebGLTensor.fromArray(arange(100, 100 + 2 * 4), [2, 4]);
      const x3 = WebGLTensor.fromArray(arange(200, 200 + 2 * 7), [2, 7]);
      const y = WebGLTensor.cat([x1, x2, x3], 1);
      assert.deepEqual(y.shape, [2, 14]);
      assert.deepEqual(
        await y.toArrayAsync(),
        [
          0, 1, 2, 100, 101, 102, 103, 200, 201, 202, 203, 204, 205, 206, 3, 4, 5,
          104, 105, 106, 107, 207, 208, 209, 210, 211, 212, 213,
        ]
      );
    });

    it('cat 3d 1', async () => {
      const x1 = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
      const x2 = WebGLTensor.fromArray(arange(100, 100 + 2 * 3 * 4), [2, 3, 4]);
      const y = WebGLTensor.cat([x1, x2]);
      assert.deepEqual(y.shape, [4, 3, 4]);
      assert.deepEqual(
        await y.toArrayAsync(),
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
          111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
        ]
      );
    });

    it('cat 3d 2', async () => {
      const x1 = WebGLTensor.fromArray(arange(2 * 3 * 4), [2, 3, 4]);
      const x2 = WebGLTensor.fromArray(arange(100, 100 + 2 * 3 * 4), [2, 3, 4]);
      const y = WebGLTensor.cat([x1, x2], 2);
      const cpu = await y.to('cpu');
      assert.deepEqual(y.shape, [2, 3, 8]);
      assert.deepEqual(cpu.get(0, 0, 7), 103);
      assert.deepEqual(cpu.get(0, 1, 0), 4);
      assert.deepEqual(cpu.get(0, 2, 0), 8);
      assert.deepEqual(cpu.get(0, 2, 7), 111);
    });
  });

  describe('split', () => {
    it('split 1', async () => {
      const x = WebGLTensor.fromArray([1, 2, 3, 4, 5], [5]);
      const y = WebGLTensor.split(x, 3);
      assert.deepEqual(y[0].shape, [3]);
      assert.deepEqual(y[1].shape, [2]);
      assert.deepEqual(await y[0].toArrayAsync(), [1, 2, 3]);
      assert.deepEqual(await y[1].toArrayAsync(), [4, 5]);
    });

    it('split 2', async () => {
      const x = WebGLTensor.fromArray([1], [1]);
      const y = WebGLTensor.split(x, 10);
      assert.deepEqual(y[0].shape, [1]);
      assert.deepEqual(await y[0].toArrayAsync(), [1]);
    });

    it('split 2d 1', async () => {
      const x = WebGLTensor.fromArray([1, 2, 3, 4], [2, 2]);
      const y = WebGLTensor.split(x, 1, 1);
      assert.deepEqual(y[0].shape, [2, 1]);
      assert.deepEqual(y[1].shape, [2, 1]);
      assert.deepEqual(await y[0].toArrayAsync(), [1, 3]);
      assert.deepEqual(await y[1].toArrayAsync(), [2, 4]);
    });

    it('split 2d 2', async () => {
      const x = WebGLTensor.fromArray([1, 2, 3, 4, 5, 6], [3, 2]);
      const y = WebGLTensor.split(x, [1, 2]);
      assert.deepEqual(y[0].shape, [1, 2]);
      assert.deepEqual(y[1].shape, [2, 2]);
      assert.deepEqual(await y[0].toArrayAsync(), [1, 2]);
      assert.deepEqual(await y[1].toArrayAsync(), [3, 4, 5, 6]);
    });

    it('split 3d 1', async () => {
      const x = WebGLTensor.fromArray(arange(100, 100 + 3 * 2 * 4), [3, 2, 4]);
      const y = WebGLTensor.split(x, [1, 2]);
      assert.deepEqual(y[0].shape, [1, 2, 4]);
      assert.deepEqual(y[1].shape, [2, 2, 4]);
      assert.deepEqual(
        await y[0].toArrayAsync(),
        [100, 101, 102, 103, 104, 105, 106, 107]
      );
      assert.deepEqual(
        await y[1].toArrayAsync(),
        [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
          122, 123]
      );
    });

    it('split 3d 2', async () => {
      const x = WebGLTensor.fromArray(arange(100, 100 + 3 * 4 * 4), [3, 4, 4]);
      const y = WebGLTensor.split(x, [1, 2, 1], 2);
      assert.deepEqual(y[0].shape, [3, 4, 1]);
      assert.deepEqual(y[1].shape, [3, 4, 2]);
      assert.deepEqual(y[2].shape, [3, 4, 1]);
      assert.deepEqual(await y[0].toArrayAsync(), [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144]);
      assert.deepEqual(
        await y[1].toArrayAsync(),
        [101, 102, 105, 106, 109, 110, 113, 114, 117, 118, 121, 122, 125, 126,
          129, 130, 133, 134, 137, 138, 141, 142, 145, 146]
      );
      assert.deepEqual(await y[2].toArrayAsync(), [103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147]);
    });

    it('split 3d 3', async () => {
      const x = WebGLTensor.fromArray(
        [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        [2, 3, 2]
      );
      const y = WebGLTensor.split(x, [1, 2], 1);
      assert.deepEqual(y[0].shape, [2, 1, 2]);
      assert.deepEqual(y[1].shape, [2, 2, 2]);
      assert.deepEqual(await y[0].toArrayAsync(), [1, 2, 1, 2]);
      assert.deepEqual(await y[1].toArrayAsync(), [3, 4, 5, 6, 3, 4, 5, 6]);
    });
  });
});

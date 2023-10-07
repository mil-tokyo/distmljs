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
});

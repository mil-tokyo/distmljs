import { assert } from 'chai';
import { WebGPUTensor } from '../../tensor/webgpu/webgpuTensor';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

describe('webgpuTensor', () => {
  if (!testFlag.webgpu) {
    return;
  }

  describe('basic', () => {
    it('computes size', async () => {
      const t = WebGPUTensor.zeros([3, 4]);
      assert.equal(t.size, 12);
      assert.equal(t.ndim, 2);
      assert.deepEqual(t.shape, [3, 4]);
      assert.deepEqual(t.strides, [4, 1]);
    });

    it('computes size of scalar', async () => {
      const t = WebGPUTensor.zeros([]);
      assert.equal(t.size, 1);
      assert.equal(t.ndim, 0);
      assert.deepEqual(t.shape, []);
      assert.deepEqual(t.strides, []);
    });

    it('create from array', async () => {
      const t = WebGPUTensor.fromArray([10, 20, 30, 4.5, 50, 60], [2, 3]);
      // WebGPUの仕様上の制限により、fromArrayでCPUから書き込んだデータを、直接読むことができない。
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 4.5);
    });

    it('create from array int32', async () => {
      // float32では正確に表せない数値
      const t = WebGPUTensor.fromArray(
        [10, 20, 30, 16843009, 16843010, 16843011],
        [2, 3],
        'int32'
      );
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 16843009);
      assert.equal(cpu.get(1, 1), 16843010);
      assert.equal(cpu.get(1, 2), 16843011);
    });

    it('create from array uint8', async () => {
      const t = WebGPUTensor.fromArray(
        [10, 20, 30, 40, 50, 60],
        [2, 3],
        'uint8'
      );
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 40);
    });

    it('create from array bool', async () => {
      const t = WebGPUTensor.fromArray([0, 1, 0, 1, 1, 0], [2, 3], 'bool');
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 1);
    });
  });

  describe('exp', () => {
    it('exp', async () => {
      const x = WebGPUTensor.fromArray([1, -1], [2]);
      const y = WebGPUTensor.exp(x);
      arrayNearlyEqual(await y.toArrayAsync(), [2.71828182845904, 0.367879441]);
    });
  });

  describe('abs', () => {
    it('abs', async () => {
      const x = WebGPUTensor.fromArray([1.5, -3.5], [2]);
      const y = WebGPUTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [1.5, 3.5]);
    });

    it('abs int32', async () => {
      const x = WebGPUTensor.fromArray([16843010, -16843009], [2], 'int32');
      const y = WebGPUTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [16843010, 16843009]);
    });
  });

  describe('add', () => {
    it('add', async () => {
      const lhs = WebGPUTensor.fromArray([10, 20], [2]);
      const rhs = WebGPUTensor.fromArray([50, 60], [2]);
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [60, 80]);
    });

    it('add int32', async () => {
      const lhs = WebGPUTensor.fromArray([16843009, 16843010], [2], 'int32');
      const rhs = WebGPUTensor.fromArray([1, -3], [2], 'int32');
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [16843010, 16843007]);
    });

    it('add uint8', async () => {
      const lhs = WebGPUTensor.fromArray([100, 101], [2], 'uint8');
      const rhs = WebGPUTensor.fromArray([1, 3], [2], 'uint8');
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [101, 104]);
    });
  });

  describe('ravel', () => {
    it('from 2d', async () => {
      const x = WebGPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGPUTensor.ravel(x);
      assert.isTrue(x.buffer.gpuBuffer === y.buffer.gpuBuffer);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.copy().toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('flatten', () => {
    it('from 2d', async () => {
      const x = WebGPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGPUTensor.flatten(x);
      assert.isFalse(x.buffer.gpuBuffer === y.buffer.gpuBuffer);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });
});

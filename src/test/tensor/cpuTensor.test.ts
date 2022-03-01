import { assert } from 'chai';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';

describe('cpuTensor', () => {
  describe('basic', () => {
    it('computes size', () => {
      const t = CPUTensor.zeros([3, 4]);
      assert.equal(t.size, 12);
      assert.equal(t.ndim, 2);
      assert.deepEqual(t.shape, [3, 4]);
      assert.deepEqual(t.strides, [4, 1]);
    });

    it('computes size of scalar', () => {
      const t = CPUTensor.zeros([]);
      assert.equal(t.size, 1);
      assert.equal(t.ndim, 0);
      assert.deepEqual(t.shape, []);
      assert.deepEqual(t.strides, []);
    });

    it('create from array', () => {
      const t = CPUTensor.fromArray([10, 20, 30, 40, 50, 60], [2, 3]);
      assert.equal(t.get(1, 0), 40);
    });
  });

  describe('broadcastShapes', () => {
    it('2d', () => {
      assert.deepEqual(
        CPUTensor.broadcastShapes([
          [1, 3],
          [2, 1],
        ]),
        [2, 3]
      );
      assert.deepEqual(CPUTensor.broadcastShapes([[3], [2, 1]]), [2, 3]);
      assert.deepEqual(CPUTensor.broadcastShapes([[3], [2, 3]]), [2, 3]);
      assert.deepEqual(CPUTensor.broadcastShapes([[], [2, 3]]), [2, 3]);
    });
    // TODO: add cases
  });

  describe('ravel', () => {
    it('from 2d', () => {
      const x = CPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = CPUTensor.ravel(x);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(y.toArray(), [0, 1, 2, 3, 4, 5]);
      // y is alias, so x will change
      y.set(10, 4);
      assert.deepEqual(y.toArray(), [0, 1, 2, 3, 10, 5]);
      assert.deepEqual(x.toArray(), [0, 1, 2, 3, 10, 5]);
    });
  });

  describe('flatten', () => {
    it('from 2d', () => {
      const x = CPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = CPUTensor.flatten(x);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(y.toArray(), [0, 1, 2, 3, 4, 5]);
      // y is copy, so x will not change
      y.set(10, 4);
      assert.deepEqual(y.toArray(), [0, 1, 2, 3, 10, 5]);
      assert.deepEqual(x.toArray(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('squeeze', () => {
    it('5d to 3d 1', () => {
      const x = CPUTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 2, 1, 2]);
      const y = CPUTensor.squeeze(x);
      assert.deepEqual(y.shape, [2, 2, 2]);
      assert.deepEqual(y.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('5d to 3d 2', () => {
      const x = CPUTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 2, 1, 2]);
      const y = CPUTensor.squeeze(x, 1);
      assert.deepEqual(y.shape, [2, 2, 1, 2]);
      assert.deepEqual(y.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
    });
  });
});

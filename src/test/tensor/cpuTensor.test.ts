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

  describe('fromArrayND', () => {
    it('from scalar number', () => {
      const x = CPUTensor.fromArrayND(3.5);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, []);
      assert.deepEqual(x.toArray(), [3.5]);
    });

    it('from scalar string', () => {
      const x = CPUTensor.fromArrayND('123.5');
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, []);
      assert.deepEqual(x.toArray(), [123.5]);
    });

    it('from scalar boolean', () => {
      const x = CPUTensor.fromArrayND(true);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, []);
      assert.deepEqual(x.toArray(), [1]);
    });

    it('from scalar number, uint8', () => {
      const x = CPUTensor.fromArrayND(5, 'uint8');
      assert.equal(x.dtype, 'uint8');
      assert.deepEqual(x.shape, []);
      assert.deepEqual(x.toArray(), [5]);
    });

    it('from 1d array', () => {
      const x = CPUTensor.fromArrayND([
        -1,
        1.5,
        '3.5',
        true,
        false,
        Infinity,
        -Infinity,
      ]);
      assert.deepEqual(x.shape, [7]);
      assert.deepEqual(x.toArray(), [-1, 1.5, 3.5, 1, 0, Infinity, -Infinity]);
    });

    it('from 1d arraylike', () => {
      const alike = { 0: 0, 1: 10, 2: 20, length: 3 };
      const x = CPUTensor.fromArrayND(alike);
      assert.deepEqual(x.shape, [3]);
      assert.deepEqual(x.toArray(), [0, 10, 20]);
    });

    it('from 1d empty array', () => {
      const x = CPUTensor.fromArrayND([]);
      assert.deepEqual(x.shape, [0]);
      assert.deepEqual(x.toArray(), []);
    });

    it('from 2d array', () => {
      const x = CPUTensor.fromArrayND([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      assert.deepEqual(x.shape, [2, 3]);
      assert.deepEqual(x.toArray(), [1, 2, 3, 4, 5, 6]);
    });

    it('from 2d array mismatch', () => {
      assert.throw(() => {
        CPUTensor.fromArrayND([
          [1, 2, 3],
          [4, 5],
        ]);
      });
    });

    it('from 2d empty array', () => {
      const x = CPUTensor.fromArrayND([[]]);
      assert.deepEqual(x.shape, [1, 0]);
      assert.deepEqual(x.toArray(), []);
    });

    it('from 2d empty array 2', () => {
      const x = CPUTensor.fromArrayND([[], []]);
      assert.deepEqual(x.shape, [2, 0]);
      assert.deepEqual(x.toArray(), []);
    });

    it('from 2d arraylike', () => {
      const alike = {
        0: { 0: 0, 1: 10, length: 2 },
        1: { 0: 100, 1: '110', length: 2 },
        length: 2,
      };
      const x = CPUTensor.fromArrayND(alike);
      assert.deepEqual(x.shape, [2, 2]);
      assert.deepEqual(x.toArray(), [0, 10, 100, 110]);
    });

    it('from 3d array', () => {
      const x = CPUTensor.fromArrayND([
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
      ]);
      assert.deepEqual(x.shape, [2, 2, 3]);
      assert.deepEqual(x.toArray(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    });

    it('from CPUTensor', () => {
      const src = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
      const x = CPUTensor.fromArrayND(src);
      assert.deepEqual(x.shape, [2, 2]);
      assert.deepEqual(x.toArray(), [1, 2, 3, 4]);
    });

    it('from array of CPUTensor', () => {
      const src0 = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
      const src1 = CPUTensor.fromArray([5, 6, 7, 8], [2, 2]);
      const src2 = CPUTensor.fromArray([9, 10, 11, 12], [2, 2]);
      const x = CPUTensor.fromArrayND([src0, src1, src2]);
      assert.deepEqual(x.shape, [3, 2, 2]);
      assert.deepEqual(x.toArray(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    });
  });

  describe('toArrayND', () => {
    it('from scalar', () => {
      const x = CPUTensor.fromArray([1], []);
      assert.deepEqual(x.toArrayND(), 1);
    });

    it('from 1d', () => {
      const x = CPUTensor.fromArray([1, 2, 3, 4], [4]);
      assert.deepEqual(x.toArrayND(), [1, 2, 3, 4]);
    });

    it('from 2d', () => {
      const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
      assert.deepEqual(x.toArrayND(), [
        [1, 2],
        [3, 4],
      ]);
    });

    it('from 3d', () => {
      const x = CPUTensor.fromArray(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [2, 2, 3]
      );
      assert.deepEqual(x.toArrayND(), [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
      ]);
    });

    it('from 1d size=0', () => {
      const x = CPUTensor.fromArray([], [0]);
      assert.deepEqual(x.toArrayND(), []); // numpyにおける、numpy.array([]).tolist()と同じ
    });

    it('from 2d size=0', () => {
      const x = CPUTensor.fromArray([], [2, 0]);
      assert.deepEqual(x.toArrayND(), [[], []]); // numpyにおける、numpy.array([[],[]]).tolist()と同じ
    });
  });
});

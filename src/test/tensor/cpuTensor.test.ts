import { assert } from 'chai';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { arange } from '../../util';

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
});

describe('sort', () => {
  it('sort 1d asc', () => {
    const x = CPUTensor.fromArray([19, 46, 7, 2], [4]);
    const [sorted, indices] = CPUTensor.sort(x);
    assert.deepEqual(sorted.shape, [4]);
    assert.deepEqual(indices.shape, [4]);
    assert.deepEqual(sorted.toArray(), [2, 7, 19, 46]);
    assert.deepEqual(indices.toArray(), [3, 2, 0, 1]);
  });
  it('sort 1d des', () => {
    const x = CPUTensor.fromArray([19, 46, 7, 2], [4]);
    const [sorted, indices] = CPUTensor.sort(x, 0, true);
    assert.deepEqual(sorted.shape, [4]);
    assert.deepEqual(indices.shape, [4]);
    assert.deepEqual(sorted.toArray(), [46, 19, 7, 2]);
    assert.deepEqual(indices.toArray(), [1, 0, 2, 3]);
  });
  it('sort 2d asc 1', () => {
    const x = CPUTensor.fromArray(
      [48, 13, 73, 24, 137, 32, 6, 84, 27, 37, 92, 55],
      [3, 4]
    );
    const [sorted, indices] = CPUTensor.sort(x, 1);
    assert.deepEqual(sorted.shape, [3, 4]);
    assert.deepEqual(indices.shape, [3, 4]);
    assert.deepEqual(
      sorted.toArray(),
      [13, 24, 48, 73, 6, 32, 84, 137, 27, 37, 55, 92]
    );
    assert.deepEqual(indices.toArray(), [1, 3, 0, 2, 2, 1, 3, 0, 0, 1, 3, 2]);
  });
  it('sort 2d asc 2', () => {
    const x = CPUTensor.fromArray(
      [48, 13, 73, 24, 137, 32, 6, 84, 27, 37, 92, 55],
      [3, 4]
    );
    const [sorted, indices] = CPUTensor.sort(x, 0);
    assert.deepEqual(sorted.shape, [3, 4]);
    assert.deepEqual(indices.shape, [3, 4]);
    assert.deepEqual(
      sorted.toArray(),
      [27, 13, 6, 24, 48, 32, 73, 55, 137, 37, 92, 84]
    );
    assert.deepEqual(indices.toArray(), [2, 0, 1, 0, 0, 1, 0, 2, 1, 2, 2, 1]);
  });
  it('sort 2d des', () => {
    const x = CPUTensor.fromArray(
      [48, 13, 73, 24, 137, 32, 6, 84, 27, 37, 92, 55],
      [3, 4]
    );
    const [sorted, indices] = CPUTensor.sort(x, 1, true);
    assert.deepEqual(sorted.shape, [3, 4]);
    assert.deepEqual(indices.shape, [3, 4]);
    assert.deepEqual(
      sorted.toArray(),
      [73, 48, 24, 13, 137, 84, 32, 6, 92, 55, 37, 27]
    );
    assert.deepEqual(indices.toArray(), [2, 0, 3, 1, 0, 3, 1, 2, 2, 3, 1, 0]);
  });
  it('sort 3d asc', () => {
    const x = CPUTensor.fromArray(
      [10, 5, 2, 9, 8, 11, 1, 3, 12, 7, 4, 6],
      [2, 2, 3]
    );
    const [sorted, indices] = CPUTensor.sort(x, 2);
    assert.deepEqual(sorted.shape, [2, 2, 3]);
    assert.deepEqual(indices.shape, [2, 2, 3]);
    assert.deepEqual(sorted.toArray(), [2, 5, 10, 8, 9, 11, 1, 3, 12, 4, 6, 7]);
    assert.deepEqual(indices.toArray(), [2, 1, 0, 1, 0, 2, 0, 1, 2, 1, 2, 0]);
  });
  it('sort negative dim', () => {
    const x = CPUTensor.fromArray(
      [48, 13, 73, 24, 137, 32, 6, 84, 27, 37, 92, 55],
      [3, 4]
    );
    const [sorted, indices] = CPUTensor.sort(x, -2);
    assert.deepEqual(sorted.shape, [3, 4]);
    assert.deepEqual(indices.shape, [3, 4]);
    assert.deepEqual(
      sorted.toArray(),
      [27, 13, 6, 24, 48, 32, 73, 55, 137, 37, 92, 84]
    );
  });
  it('stable sort', () => {
    const x = CPUTensor.fromArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [10]);
    const [sorted, indices] = CPUTensor.sort(x);
    assert.deepEqual(sorted.shape, [10]);
    assert.deepEqual(indices.shape, [10]);
    assert.deepEqual(sorted.toArray(), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    assert.deepEqual(indices.toArray(), [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]);
  });
});

describe('max', () => {
  it('max to scalar', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [4]);
    const y = CPUTensor.max(x);
    assert.deepEqual(y.shape, []);
    assert.deepEqual(y.toArray(), [202]);
  });
  it('max to array 1', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [2, 2]);
    const [y, indices] = CPUTensor.max(x, 1);
    assert.deepEqual(y.shape, [2]);
    assert.deepEqual(y.toArray(), [37, 202]);
    assert.deepEqual(indices.shape, [2]);
    assert.deepEqual(indices.toArray(), [0, 0]);
    assert.equal(indices.dtype, 'int32');
  });
  it('max to array 2', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.max(x, 2);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [202, 93, 99, 84]);
    assert.deepEqual(indices.shape, [2, 2]);
    assert.deepEqual(indices.toArray(), [2, 2, 1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
  it('max to array 2 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.max(x, 2, true);
    assert.deepEqual(y.shape, [2, 2, 1]);
    assert.deepEqual(y.toArray(), [202, 93, 99, 84]);
    assert.deepEqual(indices.shape, [2, 2, 1]);
    assert.deepEqual(indices.toArray(), [2, 2, 1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
  it('max to array 2 negative dim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.max(x, -1);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [202, 93, 99, 84]);
    assert.deepEqual(indices.shape, [2, 2]);
    assert.deepEqual(indices.toArray(), [2, 2, 1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
  it('max to array 3', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.max(x, 1);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [65, 26, 202, 62, 99, 64]);
    assert.deepEqual(indices.shape, [2, 3]);
    assert.deepEqual(indices.toArray(), [1, 1, 0, 1, 0, 0]);
    assert.equal(indices.dtype, 'int32');
  });
  it('max to array 3 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.max(x, 1, true);
    assert.deepEqual(y.shape, [2, 1, 3]);
    assert.deepEqual(y.toArray(), [65, 26, 202, 62, 99, 64]);
    assert.deepEqual(indices.shape, [2, 1, 3]);
    assert.deepEqual(indices.toArray(), [1, 1, 0, 1, 0, 0]);
    assert.equal(indices.dtype, 'int32');
  });
});

describe('min', () => {
  it('min to scalar', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [4]);
    const y = CPUTensor.min(x);
    assert.deepEqual(y.shape, []);
    assert.deepEqual(y.toArray(), [1]);
  });
  it('min to array 1', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [2, 2]);
    const [y, indices] = CPUTensor.min(x, 1);
    assert.deepEqual(y.shape, [2]);
    assert.deepEqual(y.toArray(), [1, 65]);
    assert.deepEqual(indices.shape, [2]);
    assert.deepEqual(indices.toArray(), [1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
  it('min to array 2', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.min(x, 2);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [1, 26, 28, 16]);
    assert.deepEqual(indices.shape, [2, 2]);
    assert.deepEqual(indices.toArray(), [1, 1, 0, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('min to array 2 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.min(x, 2, true);
    assert.deepEqual(y.shape, [2, 2, 1]);
    assert.deepEqual(y.toArray(), [1, 26, 28, 16]);
    assert.deepEqual(indices.shape, [2, 2, 1]);
    assert.deepEqual(indices.toArray(), [1, 1, 0, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('min to array 2 negative dim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.min(x, -1);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [1, 26, 28, 16]);
    assert.deepEqual(indices.shape, [2, 2]);
    assert.deepEqual(indices.toArray(), [1, 1, 0, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('min to array 3', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.min(x, 1);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [37, 1, 93, 28, 84, 16]);
    assert.deepEqual(indices.shape, [2, 3]);
    assert.deepEqual(indices.toArray(), [0, 0, 1, 0, 1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
  it('min to array 3 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const [y, indices] = CPUTensor.min(x, 1, true);
    assert.deepEqual(y.shape, [2, 1, 3]);
    assert.deepEqual(y.toArray(), [37, 1, 93, 28, 84, 16]);
    assert.deepEqual(indices.shape, [2, 1, 3]);
    assert.deepEqual(indices.toArray(), [0, 0, 1, 0, 1, 1]);
    assert.equal(indices.dtype, 'int32');
  });
});

describe('argmax', () => {
  it('argmax to scalar', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [4]);
    const y = CPUTensor.argmax(x);
    assert.deepEqual(y.shape, []);
    assert.deepEqual(y.toArray(), [2]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 1', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [2, 2]);
    const y = CPUTensor.argmax(x, 1);
    assert.deepEqual(y.shape, [2]);
    assert.deepEqual(y.toArray(), [0, 0]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 2', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmax(x, 2);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [2, 2, 1, 1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 2 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmax(x, 2, true);
    assert.deepEqual(y.shape, [2, 2, 1]);
    assert.deepEqual(y.toArray(), [2, 2, 1, 1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 2 negative dim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmax(x, -1);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [2, 2, 1, 1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 3', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmax(x, 1);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [1, 1, 0, 1, 0, 0]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmax to array 3 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmax(x, 1, true);
    assert.deepEqual(y.shape, [2, 1, 3]);
    assert.deepEqual(y.toArray(), [1, 1, 0, 1, 0, 0]);
    assert.equal(y.dtype, 'int32');
  });
});

describe('argmin', () => {
  it('argmin to scalar', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [4]);
    const y = CPUTensor.argmin(x);
    assert.deepEqual(y.shape, []);
    assert.deepEqual(y.toArray(), [1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 1', () => {
    const x = CPUTensor.fromArray([37, 1, 202, 65], [2, 2]);
    const y = CPUTensor.argmin(x, 1);
    assert.deepEqual(y.shape, [2]);
    assert.deepEqual(y.toArray(), [1, 1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 2', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmin(x, 2);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [1, 1, 0, 2]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 2 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmin(x, 2, true);
    assert.deepEqual(y.shape, [2, 2, 1]);
    assert.deepEqual(y.toArray(), [1, 1, 0, 2]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 2 negative dim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmin(x, -1);
    assert.deepEqual(y.shape, [2, 2]);
    assert.deepEqual(y.toArray(), [1, 1, 0, 2]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 3', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmin(x, 1);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [0, 0, 1, 0, 1, 1]);
    assert.equal(y.dtype, 'int32');
  });
  it('argmin to array 3 keepdim', () => {
    const x = CPUTensor.fromArray(
      [37, 1, 202, 65, 26, 93, 28, 99, 64, 62, 84, 16],
      [2, 2, 3]
    );
    const y = CPUTensor.argmin(x, 1, true);
    assert.deepEqual(y.shape, [2, 1, 3]);
    assert.deepEqual(y.toArray(), [0, 0, 1, 0, 1, 1]);
    assert.equal(y.dtype, 'int32');
  });
});

describe('topk', () => {
  it('topk 1d', () => {
    const x = CPUTensor.fromArray(arange(6), [6]);
    const [y, indices] = CPUTensor.topk(x, 3);
    assert.deepEqual(y.shape, [3]);
    assert.deepEqual(y.toArray(), [5, 4, 3]);
    assert.equal(y.dtype, 'float32');
    assert.deepEqual(indices.shape, [3]);
    assert.deepEqual(indices.toArray(), [5, 4, 3]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 2d 1', () => {
    const x = CPUTensor.fromArray(
      [37, 92, 23, 283, 13, 73, 26, 9, 64],
      [3, 3],
      'int32'
    );
    const [y, indices] = CPUTensor.topk(x, 2);
    assert.deepEqual(y.shape, [3, 2]);
    assert.deepEqual(y.toArray(), [92, 37, 283, 73, 64, 26]);
    assert.equal(y.dtype, 'int32');
    assert.deepEqual(indices.shape, [3, 2]);
    assert.deepEqual(indices.toArray(), [1, 0, 0, 2, 2, 0]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 2d 2', () => {
    const x = CPUTensor.fromArray(
      [37, 92, 23, 283, 13, 73, 26, 9, 64],
      [3, 3],
      'int32'
    );
    const [y, indices] = CPUTensor.topk(x, 2, -1, false);
    assert.deepEqual(y.shape, [3, 2]);
    assert.deepEqual(y.toArray(), [23, 37, 13, 73, 9, 26]);
    assert.equal(y.dtype, 'int32');
    assert.deepEqual(indices.shape, [3, 2]);
    assert.deepEqual(indices.toArray(), [2, 0, 1, 2, 1, 0]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 2d 3', () => {
    const x = CPUTensor.fromArray(
      [37, 92, 23, 283, 13, 73, 26, 9, 64],
      [3, 3],
      'int32'
    );
    const [y, indices] = CPUTensor.topk(x, 2, 0, false);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [26, 9, 23, 37, 13, 64]);
    assert.equal(y.dtype, 'int32');
    assert.deepEqual(indices.shape, [2, 3]);
    assert.deepEqual(indices.toArray(), [2, 2, 0, 0, 1, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 3d 1', () => {
    const x = CPUTensor.fromArray(arange(3 * 3 * 3), [3, 3, 3]);
    const [y, indices] = CPUTensor.topk(x, 1, 1);
    assert.deepEqual(y.shape, [3, 1, 3]);
    assert.deepEqual(y.toArray(), [6, 7, 8, 15, 16, 17, 24, 25, 26]);
    assert.equal(y.dtype, 'float32');
    assert.deepEqual(indices.shape, [3, 1, 3]);
    assert.deepEqual(indices.toArray(), [2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 3d 2', () => {
    const x = CPUTensor.fromArray(arange(3 * 3 * 3), [3, 3, 3]);
    const [y, indices] = CPUTensor.topk(x, 1, 1);
    assert.deepEqual(y.shape, [3, 1, 3]);
    assert.deepEqual(y.toArray(), [6, 7, 8, 15, 16, 17, 24, 25, 26]);
    assert.equal(y.dtype, 'float32');
    assert.deepEqual(indices.shape, [3, 1, 3]);
    assert.deepEqual(indices.toArray(), [2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert.equal(indices.dtype, 'int32');
  });
  it('topk 3d 3', () => {
    const x = CPUTensor.fromArray(arange(3 * 3 * 3), [3, 3, 3]);
    const [y, indices] = CPUTensor.topk(x, 3, 1);
    assert.deepEqual(y.shape, [3, 3, 3]);
    assert.deepEqual(
      y.toArray(),
      [
        6, 7, 8, 3, 4, 5, 0, 1, 2, 15, 16, 17, 12, 13, 14, 9, 10, 11, 24, 25,
        26, 21, 22, 23, 18, 19, 20,
      ]
    );
    assert.equal(y.dtype, 'float32');
    assert.deepEqual(indices.shape, [3, 3, 3]);
    assert.deepEqual(
      indices.toArray(),
      [
        2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1,
        0, 0, 0,
      ]
    );
    assert.equal(indices.dtype, 'int32');
  });
});

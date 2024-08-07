import { assert } from 'chai';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { arange } from '../../util';

describe('repeat', () => {
  it('repeat scalar', () => {
    const x = CPUTensor.s(3);
    const y = CPUTensor.repeat(x, 4);
    assert.deepEqual(y.shape, [4]);
    assert.deepEqual(y.toArray(), [3, 3, 3, 3]);
  });

  it('repeat without axis', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.repeat(x, 2);
    assert.deepEqual(y.shape, [8]);
    assert.deepEqual(y.toArray(), [1, 1, 2, 2, 3, 3, 4, 4]);
  });

  it('repeat axis=0', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.repeat(x, 2, 1);
    assert.deepEqual(y.shape, [2, 4]);
    assert.deepEqual(y.toArray(), [1, 1, 2, 2, 3, 3, 4, 4]);
  });

  it('repeat axis=1', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.repeat(x, 2, 0);
    assert.deepEqual(y.shape, [4, 2]);
    assert.deepEqual(y.toArray(), [1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('repeats with array1', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.repeat(x, [1, 2], 0);
    assert.deepEqual(y.shape, [3, 2]);
    assert.deepEqual(y.toArray(), [1, 2, 3, 4, 3, 4]);
  });

  it('repeats with array3', () => {
    const x = CPUTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
    const y = CPUTensor.repeat(x, [3, 2, 4, 1], 2);
    assert.deepEqual(y.shape, [2, 3, 10, 5]);
    assert.deepEqual(y.get(0, 2, 2, 0), 40);
    assert.deepEqual(y.get(1, 1, 8, 1), 91);
    assert.deepEqual(y.get(1, 1, 9, 1), 96);
  });
});

describe('tile', () => {
  it('tile scalar', () => {
    const x = CPUTensor.s(3);
    const y = CPUTensor.tile(x, 2);
    assert.deepEqual(y.shape, [2]);
    assert.deepEqual(y.toArray(), [3, 3]);
  });

  it('tile scalar 2d', () => {
    const x = CPUTensor.s(3);
    const y = CPUTensor.tile(x, [2, 3]);
    assert.deepEqual(y.shape, [2, 3]);
    assert.deepEqual(y.toArray(), [3, 3, 3, 3, 3, 3]);
  });

  it('tile 1', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.tile(x, 2);
    assert.deepEqual(y.shape, [2, 4]);
    assert.deepEqual(y.toArray(), [1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('tile 2', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.tile(x, [2, 3]);
    assert.deepEqual(y.shape, [4, 6]);
    assert.deepEqual(
      y.toArray(),
      [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    );
  });

  it('tile 3', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [2, 2]);
    const y = CPUTensor.tile(x, [2, 1, 3]);
    assert.deepEqual(y.shape, [2, 2, 6]);
    assert.deepEqual(
      y.toArray(),
      [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    );
  });

  it('tile 4', () => {
    const x = CPUTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
    const y = CPUTensor.tile(x, 2);
    assert.deepEqual(y.shape, [2, 3, 4, 10]);
    assert.deepEqual(
      y.toArray(),
      [
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 70, 71, 72, 73,
        74, 75, 76, 77, 78, 79, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 80, 81,
        82, 83, 84, 85, 86, 87, 88, 89, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 95, 96, 97, 98, 99, 100, 101,
        102, 103, 104, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 105,
        106, 107, 108, 109, 110, 111, 112, 113, 114, 110, 111, 112, 113, 114,
        115, 116, 117, 118, 119, 115, 116, 117, 118, 119,
      ]
    );
  });

  it('tile 5', () => {
    const x = CPUTensor.fromArray(arange(2 * 3 * 4 * 5), [2, 3, 4, 5]);
    const y = CPUTensor.tile(x, [2, 1]);
    assert.deepEqual(y.shape, [2, 3, 8, 5]);
    assert.deepEqual(
      y.toArray(),
      [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 60, 61, 62, 63, 64, 65, 66, 67, 68,
        69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
        102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        116, 117, 118, 119, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      ]
    );
  });
});

describe('chunk', () => {
  it('chunk 1', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4], [4]);
    const y = CPUTensor.chunk(x, 2);
    assert.deepEqual(y[0].shape, [2]);
    assert.deepEqual(y[1].shape, [2]);
    assert.deepEqual(y[0].toArray(), [1, 2]);
    assert.deepEqual(y[1].toArray(), [3, 4]);
  });

  it('chunk 2', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      [3, 4]
    );
    const y = CPUTensor.chunk(x, 3, 1);
    assert.deepEqual(y[0].shape, [3, 2]);
    assert.deepEqual(y[1].shape, [3, 2]);
    assert.deepEqual(y[0].toArray(), [1, 2, 5, 6, 9, 10]);
    assert.deepEqual(y[1].toArray(), [3, 4, 7, 8, 11, 12]);
  });

  it('chunk 3', () => {
    const x = CPUTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5]);
    const y = CPUTensor.chunk(x, 3, 1);
    assert.deepEqual(y[0].shape, [2, 2]);
    assert.deepEqual(y[1].shape, [2, 2]);
    assert.deepEqual(y[2].shape, [2, 1]);
    assert.deepEqual(y[0].toArray(), [1, 2, 6, 7]);
    assert.deepEqual(y[1].toArray(), [3, 4, 8, 9]);
    assert.deepEqual(y[2].toArray(), [5, 10]);
  });

  it('chunk 4', () => {
    const x = CPUTensor.zeros([3, 4, 4]);
    const y = CPUTensor.chunk(x, 2, 0);
    assert.deepEqual(y[0].shape, [2, 4, 4]);
    assert.deepEqual(y[1].shape, [1, 4, 4]);
    assert.deepEqual(
      y[0].toArray(),
      [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
      ]
    );
    assert.deepEqual(
      y[1].toArray(),
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
  });

  it('chunk 4', () => {
    const x = CPUTensor.fromArray([1, 2, 3], [3, 1]);
    const y = CPUTensor.chunk(x, 2);
    assert.deepEqual(y[0].shape, [2, 1]);
    assert.deepEqual(y[1].shape, [1, 1]);
    assert.deepEqual(y[0].toArray(), [1, 2]);
    assert.deepEqual(y[1].toArray(), [3]);
  });
});

describe('triu', () => {
  it('triu diagnonal=undefined', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.triu(x);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 0, 13, 14, 15, 0, 0, 0, 19, 20]
    );
  });

  it('triu diagnonal=1', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.triu(x, 1);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [0, 2, 3, 4, 5, 0, 0, 8, 9, 10, 0, 0, 0, 14, 15, 0, 0, 0, 0, 20]
    );
  });

  it('triu diagnonal=-1', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.triu(x, -1);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0, 0, 18, 19, 20]
    );
  });
});

describe('tril', () => {
  it('tril diagnonal=undefined', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.tril(x);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [1, 0, 0, 0, 0, 6, 7, 0, 0, 0, 11, 12, 13, 0, 0, 16, 17, 18, 19, 0]
    );
  });

  it('tril diagnonal=1', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.tril(x, 1);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [1, 2, 0, 0, 0, 6, 7, 8, 0, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19, 20]
    );
  });

  it('tril diagnonal=-1', () => {
    const x = CPUTensor.fromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [4, 5]
    );
    const y = CPUTensor.tril(x, -1);
    assert.deepEqual(y.shape, [4, 5]);
    assert.deepEqual(
      y.toArray(),
      [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 11, 12, 0, 0, 0, 16, 17, 18, 0, 0]
    );
  });
});

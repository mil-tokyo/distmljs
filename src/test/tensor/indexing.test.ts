import { assert } from 'chai';
import { ellipsis, newaxis, slice } from '../..';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { arange } from '../../util';

describe('cpuTensor/indexing', () => {
  describe('basic', () => {
    it('get 1d scalar', () => {
      const t = CPUTensor.fromArray(arange(10), [10]);
      // スカラーが指定された次元はsqueezeされる
      let v: CPUTensor;
      v = t.gets(1);
      assert.deepEqual(v.shape, []);
      assert.deepEqual(v.toArray(), [1]);
      v = t.gets(-3);
      assert.deepEqual(v.shape, []);
      assert.deepEqual(v.toArray(), [7]);
      // スカラーで範囲外を指定すると例外
      assert.throw(() => {
        t.gets(10);
      });
      assert.throw(() => {
        t.gets(-11);
      });
    });

    it('get 1d', () => {
      const t = CPUTensor.fromArray(arange(10), [10]);
      let v: CPUTensor;
      v = t.gets();
      assert.deepEqual(v.shape, [10]);
      assert.deepEqual(v.toArray(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      v = t.gets(slice());
      assert.deepEqual(v.shape, [10]);
      assert.deepEqual(v.toArray(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      v = t.gets(slice(1, 4));
      assert.deepEqual(v.shape, [3]);
      assert.deepEqual(v.toArray(), [1, 2, 3]);
      // sliceで長さ1となってもsqueezeされない
      v = t.gets(slice(1, 2));
      assert.deepEqual(v.shape, [1]);
      assert.deepEqual(v.toArray(), [1]);
      v = t.gets(slice(1, 7, 2));
      assert.deepEqual(v.shape, [3]);
      assert.deepEqual(v.toArray(), [1, 3, 5]);
      v = t.gets(slice(8, 15));
      assert.deepEqual(v.shape, [2]);
      assert.deepEqual(v.toArray(), [8, 9]);
      // sliceで範囲外を指定すると範囲内にクリップされる
      v = t.gets(slice(-1000, 15));
      assert.deepEqual(v.shape, [10]);
      assert.deepEqual(v.toArray(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      v = t.gets(slice(1, 7, -2));
      assert.deepEqual(v.shape, [0]);
      assert.deepEqual(v.toArray(), []);
      v = t.gets(slice(0, null, -1));
      assert.deepEqual(v.shape, [1]);
      assert.deepEqual(v.toArray(), [0]);
      v = t.gets(slice(-1, null, -1));
      assert.deepEqual(v.shape, [10]);
      assert.deepEqual(v.toArray(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
      v = t.gets(slice(null, 0, -1));
      assert.deepEqual(v.shape, [9]);
      assert.deepEqual(v.toArray(), [9, 8, 7, 6, 5, 4, 3, 2, 1]);
      v = t.gets(slice(null, null, -3));
      assert.deepEqual(v.shape, [4]);
      assert.deepEqual(v.toArray(), [9, 6, 3, 0]);
    });

    it('get 2d', () => {
      const t = CPUTensor.fromArray(arange(12), [3, 4]);
      let v: CPUTensor;
      v = t.gets();
      assert.deepEqual(v.shape, [3, 4]);
      assert.deepEqual(v.toArray(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
      v = t.gets(slice(), slice());
      assert.deepEqual(v.shape, [3, 4]);
      assert.deepEqual(v.toArray(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
      v = t.gets(slice(0, 2), slice(1, 3));
      assert.deepEqual(v.shape, [2, 2]);
      assert.deepEqual(v.toArray(), [1, 2, 5, 6]);
      v = t.gets(2, slice(1, 3));
      assert.deepEqual(v.shape, [2]);
      assert.deepEqual(v.toArray(), [9, 10]);
      v = t.gets(slice(1, 3), -1);
      assert.deepEqual(v.shape, [2]);
      assert.deepEqual(v.toArray(), [7, 11]);
      v = t.gets(slice(1, 3), slice(4, null, -2));
      assert.deepEqual(v.shape, [2, 2]);
      assert.deepEqual(v.toArray(), [7, 5, 11, 9]);
      v = t.gets(2, 1);
      assert.deepEqual(v.shape, []);
      assert.deepEqual(v.toArray(), [9]);
    });

    it('get ellipsis', () => {
      const t = CPUTensor.fromArray(arange(60), [3, 4, 5]);
      let v: CPUTensor;
      v = t.gets(1, ellipsis, 3);
      assert.deepEqual(v.shape, [4]);
      assert.deepEqual(v.toArray(), [23, 28, 33, 38]);
      v = t.gets(ellipsis, slice(2, 4));
      assert.deepEqual(v.shape, [3, 4, 2]);
      assert.deepEqual(
        v.toArray(),
        [
          2, 3, 7, 8, 12, 13, 17, 18, 22, 23, 27, 28, 32, 33, 37, 38, 42, 43,
          47, 48, 52, 53, 57, 58,
        ]
      );
      v = t.gets(ellipsis, slice(2, 4), 1, 3);
      assert.deepEqual(v.shape, [1]);
      assert.deepEqual(v.toArray(), [48]);
      assert.throw(() => {
        // ellipsisは最大1個
        t.gets(ellipsis, 1, ellipsis);
      });
    });

    it('get newaxis', () => {
      const t = CPUTensor.fromArray(arange(60), [3, 4, 5]);
      let v: CPUTensor;
      v = t.gets(newaxis, 2, 1, 1);
      assert.deepEqual(v.shape, [1]);
      assert.deepEqual(v.toArray(), [46]);
      v = t.gets(newaxis, slice(0, 2), 1, 3);
      assert.deepEqual(v.shape, [1, 2]);
      assert.deepEqual(v.toArray(), [8, 28]);
      v = t.gets(newaxis, 2, ellipsis, slice(1, 3), newaxis);
      assert.deepEqual(v.shape, [1, 4, 2, 1]);
      assert.deepEqual(v.toArray(), [41, 42, 46, 47, 51, 52, 56, 57]);
    });

    it('set 1d scalar', () => {
      const t = CPUTensor.fromArray(arange(10), [10]);
      // スカラーが指定された次元はsqueezeされる
      t.sets(100, 1);
      assert.deepEqual(t.toArray(), [0, 100, 2, 3, 4, 5, 6, 7, 8, 9]);
      t.sets(200, -3);
      assert.deepEqual(t.toArray(), [0, 100, 2, 3, 4, 5, 6, 200, 8, 9]);
      // スカラーで範囲外を指定すると例外
      assert.throw(() => {
        t.sets(123, 10);
      });
      assert.throw(() => {
        t.sets(123, -11);
      });
    });

    it('set 1d', () => {
      const t = CPUTensor.fromArray(arange(10), [10]);
      let v: CPUTensor;
      v = CPUTensor.fromArray(arange(100, 110), [10]);
      t.sets(v);
      assert.deepEqual(v.shape, [10]);
      assert.deepEqual(
        t.toArray(),
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
      );

      v = CPUTensor.fromArray(arange(200, 210), [10]);
      t.sets(v, slice());
      assert.deepEqual(
        t.toArray(),
        [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
      );

      v = CPUTensor.fromArray(arange(300, 303), [3]);
      t.sets(v, slice(1, 4));
      assert.deepEqual(
        t.toArray(),
        [200, 300, 301, 302, 204, 205, 206, 207, 208, 209]
      );

      // broadcast
      v = CPUTensor.fromArray([400], [1]);
      t.sets(v, slice(1, 4));
      assert.deepEqual(
        t.toArray(),
        [200, 400, 400, 400, 204, 205, 206, 207, 208, 209]
      );

      v = CPUTensor.fromArray(arange(500, 503), [3]);
      t.sets(v, slice(1, 7, 2));
      assert.deepEqual(
        t.toArray(),
        [200, 500, 400, 501, 204, 502, 206, 207, 208, 209]
      );

      v = CPUTensor.fromArray(arange(600, 602), [2]);
      t.sets(v, slice(8, 15));
      assert.deepEqual(
        t.toArray(),
        [200, 500, 400, 501, 204, 502, 206, 207, 600, 601]
      );

      v = CPUTensor.fromArray([], [0]);
      t.sets(v, slice(1, 7, -2));
      // no change
      assert.deepEqual(
        t.toArray(),
        [200, 500, 400, 501, 204, 502, 206, 207, 600, 601]
      );

      t.sets(3, slice(1, 7, -2));
      // no change and no error
      assert.deepEqual(
        t.toArray(),
        [200, 500, 400, 501, 204, 502, 206, 207, 600, 601]
      );

      v = CPUTensor.fromArray([700], [1]);
      t.sets(v, slice(0, null, -1));
      // no change
      assert.deepEqual(
        t.toArray(),
        [700, 500, 400, 501, 204, 502, 206, 207, 600, 601]
      );

      v = CPUTensor.fromArray(arange(800, 810), [10]);
      t.sets(v, slice(-1, null, -1));
      assert.deepEqual(
        t.toArray(),
        [809, 808, 807, 806, 805, 804, 803, 802, 801, 800]
      );

      v = CPUTensor.fromArray(arange(900, 909), [9]);
      t.sets(v, slice(null, 0, -1));
      assert.deepEqual(
        t.toArray(),
        [809, 908, 907, 906, 905, 904, 903, 902, 901, 900]
      );

      v = CPUTensor.fromArray(arange(1000, 1004), [4]);
      t.sets(v, slice(null, null, -3));
      assert.deepEqual(
        t.toArray(),
        [1003, 908, 907, 1002, 905, 904, 1001, 902, 901, 1000]
      );
    });

    it('set 2d', () => {
      const t = CPUTensor.fromArray(arange(12), [3, 4]);
      let v: CPUTensor;

      v = CPUTensor.fromArray(arange(100, 112), [3, 4]);
      t.sets(v);
      assert.deepEqual(
        t.toArray(),
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
      );

      v = CPUTensor.fromArray(arange(200, 212), [3, 4]);
      t.sets(v, slice(), slice());
      assert.deepEqual(
        t.toArray(),
        [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]
      );

      v = CPUTensor.fromArray(arange(300, 304), [2, 2]);
      t.sets(v, slice(0, 2), slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 300, 301, 203, 204, 302, 303, 207, 208, 209, 210, 211]
      );

      t.sets(50, slice(0, 2), slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 50, 50, 203, 204, 50, 50, 207, 208, 209, 210, 211]
      );

      // broadcast
      v = CPUTensor.fromArray(arange(500, 502), [2, 1]);
      t.sets(v, slice(0, 2), slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 500, 500, 203, 204, 501, 501, 207, 208, 209, 210, 211]
      );

      v = CPUTensor.fromArray(arange(600, 602), [1, 2]);
      t.sets(v, slice(0, 2), slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 600, 601, 203, 204, 600, 601, 207, 208, 209, 210, 211]
      );

      v = CPUTensor.fromArray(arange(700, 702), [2]);
      t.sets(v, slice(0, 2), slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 700, 701, 207, 208, 209, 210, 211]
      );

      v = CPUTensor.fromArray(arange(800, 802), [2]);
      t.sets(v, 2, slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 700, 701, 207, 208, 800, 801, 211]
      );

      // head dimension of 1 is ok
      v = CPUTensor.fromArray(arange(900, 902), [1, 1, 2]);
      t.sets(v, 2, slice(1, 3));
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 700, 701, 207, 208, 900, 901, 211]
      );

      t.sets(1000, slice(1, 3), -1);
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 700, 701, 1000, 208, 900, 901, 1000]
      );

      v = CPUTensor.fromArray(arange(1100, 1104), [2, 2]);
      t.sets(v, slice(1, 3), slice(4, null, -2));
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 1101, 701, 1100, 208, 1103, 901, 1102]
      );

      v = CPUTensor.fromArray([1200], []);
      t.sets(v, 2, 1);
      assert.deepEqual(
        t.toArray(),
        [200, 700, 701, 203, 204, 1101, 701, 1100, 208, 1200, 901, 1102]
      );
    });

    it('set ellipsis', () => {
      const t = CPUTensor.fromArray(arange(60), [3, 4, 5]);
      let v: CPUTensor;
      v = CPUTensor.fromArray(arange(100, 104), [4]);
      t.sets(v, 1, ellipsis, 3);
      assert.deepEqual(
        t.toArray(),
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 100, 24, 25, 26, 27, 101, 29, 30, 31, 32, 102, 34, 35, 36,
          37, 103, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
          54, 55, 56, 57, 58, 59,
        ]
      );

      v = CPUTensor.fromArray(arange(200, 206), [3, 1, 2]);
      t.sets(v, ellipsis, slice(2, 4));
      assert.deepEqual(
        t.toArray(),
        [
          0, 1, 200, 201, 4, 5, 6, 200, 201, 9, 10, 11, 200, 201, 14, 15, 16,
          200, 201, 19, 20, 21, 202, 203, 24, 25, 26, 202, 203, 29, 30, 31, 202,
          203, 34, 35, 36, 202, 203, 39, 40, 41, 204, 205, 44, 45, 46, 204, 205,
          49, 50, 51, 204, 205, 54, 55, 56, 204, 205, 59,
        ]
      );
      assert.throw(() => {
        // broadcast shape mismatch
        v = CPUTensor.fromArray(arange(200, 206), [3, 2]);
        t.sets(v, ellipsis, slice(2, 4));
      });

      t.sets(300, ellipsis, slice(2, 4), 1, 3);
      assert.deepEqual(
        t.toArray(),
        [
          0, 1, 200, 201, 4, 5, 6, 200, 201, 9, 10, 11, 200, 201, 14, 15, 16,
          200, 201, 19, 20, 21, 202, 203, 24, 25, 26, 202, 203, 29, 30, 31, 202,
          203, 34, 35, 36, 202, 203, 39, 40, 41, 204, 205, 44, 45, 46, 204, 300,
          49, 50, 51, 204, 205, 54, 55, 56, 204, 205, 59,
        ]
      );
      assert.throw(() => {
        // ellipsisは最大1個
        t.gets(ellipsis, 1, ellipsis);
      });
    });

    it('set newaxis', () => {
      const t = CPUTensor.fromArray(arange(60), [3, 4, 5]);
      let v: CPUTensor;
      v = CPUTensor.fromArray(arange(100, 102), [2, 1]);
      t.sets(v, slice(0, 2), newaxis, 1, 3);
      assert.deepEqual(
        t.toArray(),
        [
          0, 1, 2, 3, 4, 5, 6, 7, 100, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 101, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
          53, 54, 55, 56, 57, 58, 59,
        ]
      );

      assert.throw(() => {
        // newaxisがなければ形状が合わない
        t.sets(v, slice(0, 2), 1, 3);
      });

      v = CPUTensor.fromArray(arange(200, 204), [4, 1, 1]);
      t.sets(v, newaxis, 2, ellipsis, slice(1, 3), newaxis);
      assert.deepEqual(
        t.toArray(),
        [
          0, 1, 2, 3, 4, 5, 6, 7, 100, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 101, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 200, 200, 43, 44, 45, 201, 201, 48, 49, 50, 202,
          202, 53, 54, 55, 203, 203, 58, 59,
        ]
      );
    });
  });
  // TODO: advanced indexing
});

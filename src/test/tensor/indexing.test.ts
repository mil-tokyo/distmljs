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
  });
  // TODO: set
  // TODO: advanced indexing
});

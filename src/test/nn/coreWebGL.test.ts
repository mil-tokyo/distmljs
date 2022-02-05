/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import { add, mul } from '../../nn/functions';
import { WebGLTensor } from '../../tensor';
import { testFlag } from '../testFlag';

async function ta(tensor: unknown): Promise<number[]> {
  assert.instanceOf(tensor, WebGLTensor);
  return await (tensor as WebGLTensor).toArrayAsync();
}

describe('nn/core/webgl', () => {
  if (!testFlag.webgl) {
    return;
  }
  describe('backprop', () => {
    it('backprop of add', async () => {
      const lhs = new Variable(WebGLTensor.fromArray([10]));
      const rhs = new Variable(WebGLTensor.fromArray([20]));
      const y = await add(lhs, rhs);
      await y.backward();
      assert.instanceOf(y.data, WebGLTensor);
      assert.deepEqual(await ta(lhs.grad!.data), [1]);
      assert.deepEqual(await ta(rhs.grad!.data), [1]);
    });

    it('backprop of add broadcast', async () => {
      const lhs = new Variable(WebGLTensor.fromArray([1, 2], [2]));
      const rhs = new Variable(WebGLTensor.fromArray([20], [1]));
      const y = await add(lhs, rhs);
      await y.backward();
      assert.deepEqual(await ta(lhs.grad!.data), [1, 1]);
      assert.deepEqual(await ta(rhs.grad!.data), [2]);
    });

    it('backprop of mul', async () => {
      const lhs = new Variable(WebGLTensor.fromArray([10]));
      const rhs = new Variable(WebGLTensor.fromArray([20]));
      const y = await mul(lhs, rhs);
      await y.backward();
      assert.deepEqual(await ta(lhs.grad!.data), [20]);
      assert.deepEqual(await ta(rhs.grad!.data), [10]);
    });
  });
});

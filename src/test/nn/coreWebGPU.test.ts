/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import { add, mul } from '../../nn/functions';
import { WebGPUTensor } from '../../tensor/webgpu/webgpuTensor';
import { testFlag } from '../testFlag';

async function ta(tensor: unknown): Promise<number[]> {
  assert.instanceOf(tensor, WebGPUTensor);
  return await (tensor as WebGPUTensor).toArrayAsync();
}

describe('nn/core/webgpu', () => {
  if (!testFlag.webgpu) {
    return;
  }
  describe('backprop', () => {
    it('backprop of add', async () => {
      const lhs = new Variable(WebGPUTensor.fromArray([10]));
      const rhs = new Variable(WebGPUTensor.fromArray([20]));
      const y = await add(lhs, rhs);
      await y.backward();
      assert.instanceOf(y.data, WebGPUTensor);
      // CPU側から1を代入したテンソルがそのまま返るため、copyがないと読み取れない(WebGPUの制約)
      assert.deepEqual(await ta(lhs.grad!.data.copy()), [1]);
      assert.deepEqual(await ta(rhs.grad!.data.copy()), [1]);
    });

    it('backprop of add broadcast', async () => {
      const lhs = new Variable(WebGPUTensor.fromArray([1, 2], [2]));
      const rhs = new Variable(WebGPUTensor.fromArray([20], [1]));
      const y = await add(lhs, rhs);
      await y.backward();
      assert.deepEqual(await ta(lhs.grad!.data.copy()), [1, 1]);
      assert.deepEqual(await ta(rhs.grad!.data.copy()), [2]);
    });

    it('backprop of mul', async () => {
      const lhs = new Variable(WebGPUTensor.fromArray([10]));
      const rhs = new Variable(WebGPUTensor.fromArray([20]));
      const y = await mul(lhs, rhs);
      await y.backward();
      assert.deepEqual(await ta(lhs.grad!.data), [20]);
      assert.deepEqual(await ta(rhs.grad!.data), [10]);
    });
  });
});
/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import { add, mul } from '../../nn/functions';
import { CPUTensor } from '../../tensor/cpuTensor';

describe('backprop', () => {
  it('backprop of add', async () => {
    const lhs = new Variable(CPUTensor.fromArray([10]));
    const rhs = new Variable(CPUTensor.fromArray([20]));
    const y = await add(lhs, rhs);
    await y.backward();
    assert.deepEqual((lhs.grad!.data as CPUTensor).toArray(), [1]);
    assert.deepEqual((rhs.grad!.data as CPUTensor).toArray(), [1]);
  });

  it('backprop of add broadcast', async () => {
    const lhs = new Variable(CPUTensor.fromArray([1, 2], [2]));
    const rhs = new Variable(CPUTensor.fromArray([20], [1]));
    const y = await add(lhs, rhs);
    await y.backward();
    assert.deepEqual((lhs.grad!.data as CPUTensor).toArray(), [1, 1]);
    assert.deepEqual((rhs.grad!.data as CPUTensor).toArray(), [2]);
  });

  it('backprop of mul', async () => {
    const lhs = new Variable(CPUTensor.fromArray([10]));
    const rhs = new Variable(CPUTensor.fromArray([20]));
    const y = await mul(lhs, rhs);
    await y.backward();
    assert.deepEqual((lhs.grad!.data as CPUTensor).toArray(), [20]);
    assert.deepEqual((rhs.grad!.data as CPUTensor).toArray(), [10]);
  });
});

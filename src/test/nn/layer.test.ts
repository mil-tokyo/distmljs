/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Backend } from '../../backend';
import { Variable } from '../../nn/core';
import { sum } from '../../nn/functions';
import { Linear } from '../../nn/layers';
import { Tensor, WebGPUTensor } from '../../tensor';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGLTensor } from '../../tensor/webgl/webglTensor';
import { arange } from '../../util';
import { testFlag } from '../testFlag';

for (const { backend, ctor } of [
  { backend: 'cpu' as Backend, ctor: CPUTensor },
  { backend: 'webgl' as Backend, ctor: WebGLTensor },
  { backend: 'webgpu' as Backend, ctor: WebGPUTensor },
]) {
  if (backend === 'webgl' && !testFlag.webgl) {
    continue;
  }
  if (backend === 'webgpu' && !testFlag.webgpu) {
    continue;
  }
  const ta = async (tensor: unknown): Promise<number[]> => {
    assert.instanceOf(tensor, ctor);
    return await (tensor as Tensor).toArrayAsync();
  };
  describe(`nn/layer/${backend}`, () => {
    describe('linear', () => {
      it('forward / backward', async () => {
        const linear = new Linear(4, 2, true);
        assert.deepEqual(linear.weight.data.shape, [2, 4]);
        assert.deepEqual(linear.bias!.data.shape, [2]);
        (linear.weight.data as CPUTensor).setArray([1, 2, 3, 4, 5, 6, 7, 8]);
        (linear.bias!.data as CPUTensor).setArray([10, 20]);
        await linear.to(backend);

        const x = new Variable(ctor.fromArray(arange(8), [2, 4]));
        const y = await linear.c(x);
        assert.deepEqual(await ta(y.data), [30, 64, 70, 168]);
        const z = await sum(y);
        await z.backward();
        assert.deepEqual(
          await ta(linear.weight.grad!.data),
          [4, 6, 8, 10, 4, 6, 8, 10]
        );
        assert.deepEqual(await ta(linear.bias!.grad!.data), [2, 2]);
      });
    });
  });
}

/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Backend } from '../../backend';
import { Variable } from '../../nn/core';
import { sum } from '../../nn/functions';
import { Linear } from '../../nn/layers';
import { SGD } from '../../nn/optimizers';
import { Tensor, WebGPUTensor } from '../../tensor';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGLTensor } from '../../tensor/webgl/webglTensor';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

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
  describe(`nn/optimizer/${backend}`, () => {
    describe('sgd', () => {
      it('forward / backward', async () => {
        const linear = new Linear(4, 2, true);
        (linear.weight.data as CPUTensor).setArray([
          -0.0037, 0.2682, -0.4115, -0.368, -0.1926, 0.1341, -0.0099, 0.3964,
        ]);
        (linear.bias!.data as CPUTensor).setArray([-0.0444, 0.1323]);
        await linear.to(backend);
        const opt = new SGD(linear.parameters(), 0.1, 0.0);
        opt.zeroGrad();
        let x = new Variable(
          ctor.fromArray(
            [
              0.3489, 0.4017, 0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8,
              0.161, 0.2823, 0.6816, 0.9152,
            ],
            [3, 4]
          )
        );
        let y = await linear.c(x);
        let z = await sum(y);
        await z.backward();
        await opt.step();
        arrayNearlyEqual(
          await ta(linear.weight.data),
          [-0.0841, 0.148, -0.5517, -0.5564, -0.273, 0.0138, -0.1501, 0.208]
        );
        arrayNearlyEqual(await ta(linear.bias!.data), [-0.3444, -0.1677]);

        opt.zeroGrad();
        x = new Variable(
          ctor.fromArray(
            [
              0.3971, 0.8742, 0.4194, 0.5529, 0.9527, 0.0362, 0.1852, 0.3734,
              0.3051, 0.932, 0.1759, 0.2698,
            ],
            [3, 4]
          )
        );
        y = await linear.c(x);
        z = await sum(y);
        await z.backward();
        await opt.step();
        arrayNearlyEqual(
          await ta(linear.weight.data),
          [-0.2496, -0.0363, -0.6297, -0.676, -0.4385, -0.1704, -0.2281, 0.0884]
        );
        arrayNearlyEqual(await ta(linear.bias!.data), [-0.6444, -0.4677]);
      });
    });

    describe('momentumsgd', () => {
      it('forward / backward', async () => {
        const linear = new Linear(4, 2, true);
        (linear.weight.data as CPUTensor).setArray([
          -0.0037, 0.2682, -0.4115, -0.368, -0.1926, 0.1341, -0.0099, 0.3964,
        ]);
        (linear.bias!.data as CPUTensor).setArray([-0.0444, 0.1323]);
        await linear.to(backend);
        const opt = new SGD(linear.parameters(), 0.1, 0.9);
        opt.zeroGrad();
        let x = new Variable(
          ctor.fromArray(
            [
              0.3489, 0.4017, 0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8,
              0.161, 0.2823, 0.6816, 0.9152,
            ],
            [3, 4]
          )
        );
        let y = await linear.c(x);
        let z = await sum(y);
        await z.backward();
        await opt.step();
        arrayNearlyEqual(
          await ta(linear.weight.data),
          [-0.0841, 0.148, -0.5517, -0.5564, -0.273, 0.0138, -0.1501, 0.208]
        );
        arrayNearlyEqual(await ta(linear.bias!.data), [-0.3444, -0.1677]);

        opt.zeroGrad();
        x = new Variable(
          ctor.fromArray(
            [
              0.3971, 0.8742, 0.4194, 0.5529, 0.9527, 0.0362, 0.1852, 0.3734,
              0.3051, 0.932, 0.1759, 0.2698,
            ],
            [3, 4]
          )
        );
        y = await linear.c(x);
        z = await sum(y);
        await z.backward();
        await opt.step();
        arrayNearlyEqual(
          await ta(linear.weight.data),
          [
            -0.322, -0.1445, -0.7559, -0.8456, -0.5108, -0.2786, -0.3543,
            -0.0811,
          ]
        );
        arrayNearlyEqual(await ta(linear.bias!.data), [-0.9144, -0.7377]);
      });
    });
  });
}

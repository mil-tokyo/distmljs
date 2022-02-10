/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import {
  linear,
  matmul,
  mseLoss,
  mul,
  relu,
  reshape,
  sigmoid,
  softmax,
  softmaxCrossEntropy,
  sub,
  sum,
  transpose,
} from '../../nn/functions';
import { Tensor, WebGPUTensor } from '../../tensor';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGLTensor } from '../../tensor/webgl/webglTensor';
import { arange } from '../../util';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

for (const { backend, ctor } of [
  { backend: 'cpu', ctor: CPUTensor },
  { backend: 'webgl', ctor: WebGLTensor },
  { backend: 'webgpu', ctor: WebGPUTensor },
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
  describe(`nn/function/${backend}`, () => {
    describe('sub', () => {
      it('backprop of sub broadcast', async () => {
        const lhs = new Variable(ctor.fromArray([1, 2], [2]));
        const rhs = new Variable(ctor.fromArray([20], [1]));
        const y = await sub(lhs, rhs);
        assert.deepEqual(await ta(y.data), [-19, -18]);
        await y.backward();
        assert.deepEqual(await ta(lhs.grad!.data.copy()), [1, 1]);
        assert.deepEqual(await ta(rhs.grad!.data.copy()), [-2]);
      });
    });

    describe('sum', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray([10, -20]));
        const y = await sum(x);
        await y.backward();
        assert.deepEqual(await ta(y.data), [-10]);
        assert.deepEqual(await ta(x.grad!.data), [1, 1]);
      });
    });

    describe('relu', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray([10, -20]));
        const y = await relu(x);
        await y.backward();
        assert.deepEqual(await ta(y.data), [10, 0]);
        assert.deepEqual(await ta(x.grad!.data), [1, 0]);
      });
    });

    describe('sigmoid', () => {
      it('backprop of sigmoid', async () => {
        const x = new Variable(ctor.fromArray([2, 3]));
        const y = await sigmoid(x);
        const z = await sum(y);
        arrayNearlyEqual(await ta(y.data), [0.8808, 0.9526]);
        await z.backward();
        arrayNearlyEqual(await ta(x.grad!.data), [0.105, 0.0452]);
      });
    });

    describe('matmul', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]));
        const y = new Variable(
          ctor.fromArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4])
        );
        const z = await matmul(x, y);
        await z.backward();
        assert.deepEqual(z.data.shape, [2, 4]);
        assert.deepEqual(await ta(z.data), [20, 23, 26, 29, 56, 68, 80, 92]);
        assert.deepEqual(await ta(x.grad!.data), [6, 22, 38, 6, 22, 38]);
        assert.deepEqual(
          await ta(y.grad!.data),
          [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
        );
      });
    });

    describe('softmax', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const z = await softmax(x);
        arrayNearlyEqual(
          await ta(z.data),
          [0.0321, 0.0871, 0.2369, 0.6439, 0.0045, 0.2436, 0.0896, 0.6623]
        );
      });
    });

    describe('softmaxCrossEntropy', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const label = new Variable(ctor.fromArray([2, 1], [2], 'int32'));
        const z = await softmaxCrossEntropy(x, label);
        await z.backward();
        arrayNearlyEqual(await ta(z.data), [1.4261]);
        arrayNearlyEqual(
          await ta(x.grad!.data),
          [0.016, 0.0436, -0.3816, 0.322, 0.0022, -0.3782, 0.0448, 0.3311]
        );
      });
    });

    describe('mseLoss', () => {
      it('backprop of mseLoss', async () => {
        const lhs = new Variable(ctor.fromArray([2, 3]));
        const rhs = new Variable(ctor.fromArray([10, 15]));
        const y = await mseLoss(lhs, rhs);
        assert.deepEqual(await ta(y.data), [104]);
        await y.backward();
        assert.deepEqual(await ta(lhs.grad!.data), [-8, -12]);
        assert.deepEqual(await ta(rhs.grad!.data), [8, 12]);
      });
    });

    describe('linear', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const weight = new Variable(ctor.fromArray(arange(12), [3, 4]));
        const bias = new Variable(ctor.fromArray(arange(3)));
        const y = await linear(x, weight, bias);
        const p = new Variable(ctor.fromArray([2, 3, -1, 5, -2, 4], [2, 3]));
        const z = await mul(y, p);
        const s = await sum(z);
        await s.backward();
        assert.deepEqual(await ta(y.data), [14, 39, 64, 19, 52, 85]);
        assert.deepEqual(await ta(s.data), [412]);
        assert.deepEqual(
          await ta(weight.grad!.data),
          [-5, 17, 14, 26, 2, -3, 2, 1, -4, 11, 6, 13]
        );
        assert.deepEqual(await ta(bias.grad!.data), [7, 1, 3]);
        assert.deepEqual(
          await ta(x.grad!.data),
          [4, 8, 12, 16, 24, 31, 38, 45]
        );
      });
    });

    describe('reshape', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const xr = await reshape(x, [4, 2]);
        const weight = new Variable(ctor.fromArray(arange(8), [4, 2]));
        const y = await mul(xr, weight);
        const s = await sum(y);
        await s.backward();
        assert.deepEqual(await ta(x.grad!.data), [0, 1, 2, 3, 4, 5, 6, 7]);
      });
    });

    describe('transpose', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const xr = await transpose(x, [1, 0]);
        const weight = new Variable(ctor.fromArray(arange(8), [4, 2]));
        const y = await mul(xr, weight);
        const s = await sum(y);
        await s.backward();
        assert.deepEqual(await ta(x.grad!.data), [0, 2, 4, 6, 1, 3, 5, 7]);
      });
    });
  });
}

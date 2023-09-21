/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import {
  bmm,
  flatten,
  linear,
  matmul,
  mseLoss,
  mul,
  log,
  sqrt,
  relu,
  reshape,
  sigmoid,
  tanh,
  softplus,
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

    describe('log', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray([10, 0.4]));
        const y = await log(x);
        const z = await sum(y);
        await z.backward();
        arrayNearlyEqual(await ta(y.data), [2.3026, -0.9163]);
        arrayNearlyEqual(await ta(x.grad!.data), [0.1, 2.5]);
      });
    });

    describe('sqrt', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray([10, 0.4]));
        const y = await sqrt(x);
        const z = await sum(y);
        await z.backward();
        arrayNearlyEqual(await ta(y.data), [3.1623, 0.6325]);
        arrayNearlyEqual(await ta(x.grad!.data), [0.1581, 0.7906]);
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

    describe('tanh', () => {
      it('backprop of tanh', async () => {
        const x = new Variable(ctor.fromArray([2, 3]));
        const y = await tanh(x);
        const z = await sum(y);
        arrayNearlyEqual(await ta(y.data), [0.9640, 0.9951]);
        await z.backward();
        arrayNearlyEqual(await ta(x.grad!.data), [0.0707, 0.0099]);
      });
    });

    describe('softplus', () => {
      it('backprop of softplus', async () => {
        const x = new Variable(ctor.fromArray([2, 3, -5]));
        const y = await softplus(x);
        const z = await sum(y);
        arrayNearlyEqual(await ta(y.data), [2.1269, 3.0486, 0.0067]);
        await z.backward();
        arrayNearlyEqual(await ta(x.grad!.data), [0.8808, 0.9526, 0.0067]);
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

    if (backend === 'cpu') {
      describe('bmm', () => {
        it('forward, backward', async () => {
          const x = new Variable(
            ctor.fromArray(
              [
                5.0, 2.0, -4.0, -2.0, 3.0, -2.0, -1.0, -5.0, 1.0, 3.0, 4.0,
                -1.0, -3.0, 0.0, 3.0, 1.0, 0.0, 5.0, 5.0, 5.0, -3.0, -2.0, 3.0,
                -2.0,
              ],
              [2, 3, 4]
            )
          );
          const y = new Variable(
            ctor.fromArray(
              [
                -4.0, -4.0, 4.0, 5.0, 4.0, 4.0, -5.0, -1.0, 3.0, -2.0, 4.0,
                -5.0, -2.0, 3.0, -4.0, -2.0, 3.0, 5.0, 4.0, 2.0, 4.0, 2.0, 5.0,
                -1.0, 3.0, 5.0, -2.0, 2.0, 3.0, 5.0, 2.0, 4.0, -2.0, 0.0, 5.0,
                3.0, 0.0, 5.0, -2.0, -2.0,
              ],
              [2, 4, 5]
            )
          );
          const z = await bmm(x, y);
          arrayNearlyEqual(
            await ta(z.data),
            [
              -24.0, -16.0, 16.0, 11.0, 28.0, -14.0, -12.0, -9.0, -14.0, 10.0,
              26.0, -42.0, -12.0, 22.0, -20.0, -3.0, 6.0, -16.0, 1.0, 4.0, 50.0,
              10.0, 25.0, 5.0, 40.0, -22.0, 10.0, -35.0, 1.0, 0.0,
            ]
          );
          const s = await sum(z);
          await s.backward();
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              5.0, -1.0, -4.0, 12.0, 5.0, -1.0, -4.0, 12.0, 5.0, -1.0, -4.0,
              12.0, 13.0, 13.0, 9.0, 4.0, 13.0, 13.0, 9.0, 4.0, 13.0, 13.0, 9.0,
              4.0,
            ]
          );
          arrayNearlyEqual(
            await ta(y.grad!.data),
            [
              9.0, 9.0, 9.0, 9.0, 9.0, 3.0, 3.0, 3.0, 3.0, 3.0, -1.0, -1.0,
              -1.0, -1.0, -1.0, -8.0, -8.0, -8.0, -8.0, -8.0, -6.0, -6.0, -6.0,
              -6.0, -6.0, 3.0, 3.0, 3.0, 3.0, 3.0, 11.0, 11.0, 11.0, 11.0, 11.0,
              4.0, 4.0, 4.0, 4.0, 4.0,
            ]
          );
        });

        it('forward, backward transa', async () => {
          const x = new Variable(
            ctor.fromArray(
              [
                -2.0, -4.0, 3.0, -1.0, 5.0, 5.0, -5.0, 0.0, -1.0, -3.0, 4.0,
                -4.0, 1.0, 2.0, -2.0, -2.0, 3.0, 0.0, 1.0, 0.0, 3.0, -4.0, 4.0,
                4.0,
              ],
              [2, 4, 3]
            )
          );
          const y = new Variable(
            ctor.fromArray(
              [
                -5.0, -3.0, 2.0, -4.0, 0.0, -1.0, 3.0, -5.0, -5.0, 5.0, -3.0,
                -3.0, 2.0, 4.0, 4.0, -2.0, 0.0, -1.0, -4.0, -4.0, -5.0, 4.0,
                -2.0, -5.0, 2.0, -4.0, 2.0, -5.0, -1.0, 5.0, 4.0, -2.0, 0.0,
                -5.0, 3.0, -1.0, -2.0, 5.0, 5.0, -5.0,
              ],
              [2, 4, 5]
            )
          );
          const z = await bmm(x, y, true, false);
          const s = await sum(z);
          await s.backward();
          arrayNearlyEqual(
            await ta(z.data),
            [
              32.0, 18.0, -6.0, 5.0, -13.0, 7.0, 27.0, -37.0, -25.0, 9.0, -9.0,
              9.0, -17.0, -25.0, 37.0, 11.0, 6.0, -12.0, -28.0, 15.0, -26.0,
              6.0, 1.0, 7.0, -1.0, 18.0, -22.0, 24.0, 15.0, -15.0,
            ]
          );
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              -10.0, -10.0, -10.0, -3.0, -3.0, -3.0, 4.0, 4.0, 4.0, -11.0,
              -11.0, -11.0, -6.0, -6.0, -6.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0,
              2.0, 2.0, 2.0,
            ]
          );
          arrayNearlyEqual(
            await ta(y.grad!.data),
            [
              -3.0, -3.0, -3.0, -3.0, -3.0, 9.0, 9.0, 9.0, 9.0, 9.0, -6.0, -6.0,
              -6.0, -6.0, -6.0, -3.0, -3.0, -3.0, -3.0, -3.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
              4.0, 4.0, 4.0, 4.0,
            ]
          );
        });

        it('forward, backward transb', async () => {
          const x = new Variable(
            ctor.fromArray(
              [
                5.0, -2.0, 5.0, -1.0, -3.0, 3.0, 0.0, 5.0, -2.0, 1.0, 4.0, -3.0,
                5.0, -1.0, -1.0, 5.0, 2.0, 2.0, -4.0, -3.0, -3.0, -1.0, -4.0,
                3.0,
              ],
              [2, 3, 4]
            )
          );
          const y = new Variable(
            ctor.fromArray(
              [
                4.0, 1.0, -2.0, 2.0, -5.0, -5.0, 5.0, -2.0, -4.0, 1.0, -1.0,
                -4.0, 3.0, 4.0, 1.0, -1.0, 3.0, -5.0, -2.0, 0.0, 4.0, -4.0,
                -2.0, -1.0, -2.0, -4.0, -5.0, 3.0, 0.0, -2.0, -3.0, -2.0, -5.0,
                3.0, 2.0, -3.0, -3.0, -5.0, 5.0, -3.0,
              ],
              [2, 5, 4]
            )
          );
          const z = await bmm(x, y, false, true);
          const s = await sum(z);
          await s.backward();
          arrayNearlyEqual(
            await ta(z.data),
            [
              6.0, 12.0, -23.0, 13.0, 15.0, 1.0, -10.0, -5.0, -2.0, -24.0,
              -21.0, 31.0, 17.0, 5.0, -19.0, 21.0, 14.0, -5.0, -45.0, -30.0,
              11.0, -1.0, 14.0, -3.0, -27.0, -3.0, 39.0, 8.0, -5.0, -15.0,
            ]
          );
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              1.0, -4.0, 1.0, -5.0, 1.0, -4.0, 1.0, -5.0, 1.0, -4.0, 1.0, -5.0,
              -6.0, -12.0, -3.0, -6.0, -6.0, -12.0, -3.0, -6.0, -6.0, -12.0,
              -3.0, -6.0,
            ]
          );
          arrayNearlyEqual(
            await ta(y.grad!.data),
            [
              0.0, 2.0, 9.0, 1.0, 0.0, 2.0, 9.0, 1.0, 0.0, 2.0, 9.0, 1.0, 0.0,
              2.0, 9.0, 1.0, 0.0, 2.0, 9.0, 1.0, 4.0, 0.0, -9.0, 5.0, 4.0, 0.0,
              -9.0, 5.0, 4.0, 0.0, -9.0, 5.0, 4.0, 0.0, -9.0, 5.0, 4.0, 0.0,
              -9.0, 5.0,
            ]
          );
        });

        it('forward, backward transa, transb', async () => {
          const x = new Variable(
            ctor.fromArray(
              [
                5.0, 3.0, 3.0, -3.0, 0.0, 0.0, -1.0, 3.0, 1.0, 2.0, 5.0, 4.0,
                -2.0, 2.0, -2.0, 4.0, 1.0, 1.0, 3.0, 1.0, -4.0, -5.0, 3.0, -1.0,
              ],
              [2, 4, 3]
            )
          );
          const y = new Variable(
            ctor.fromArray(
              [
                0.0, -1.0, 3.0, 3.0, -4.0, 3.0, 1.0, 5.0, 3.0, 0.0, 0.0, 3.0,
                -2.0, 5.0, -3.0, 3.0, -3.0, -1.0, -1.0, -2.0, 1.0, 1.0, 4.0,
                -4.0, 1.0, 1.0, 2.0, -1.0, 0.0, -1.0, 4.0, -5.0, -1.0, 2.0,
                -5.0, 0.0, -1.0, -3.0, -3.0, 1.0,
              ],
              [2, 5, 4]
            )
          );
          const z = await bmm(x, y, true, true);
          const s = await sum(z);
          await s.backward();
          arrayNearlyEqual(
            await ta(z.data),
            [
              6.0, -20.0, 21.0, -16.0, -15.0, 24.0, 16.0, 24.0, 0.0, -22.0,
              15.0, 9.0, 21.0, 3.0, -18.0, 34.0, 13.0, 33.0, -5.0, -24.0, -5.0,
              2.0, -12.0, -5.0, -5.0, -13.0, -8.0, -12.0, 24.0, 10.0,
            ]
          );
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              -6.0, -6.0, -6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 12.0, 12.0, 12.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, -9.0, -9.0, -9.0,
            ]
          );
          arrayNearlyEqual(
            await ta(y.grad!.data),
            [
              11.0, -3.0, 3.0, 11.0, 11.0, -3.0, 3.0, 11.0, 11.0, -3.0, 3.0,
              11.0, 11.0, -3.0, 3.0, 11.0, 11.0, -3.0, 3.0, 11.0, -2.0, 6.0,
              0.0, -3.0, -2.0, 6.0, 0.0, -3.0, -2.0, 6.0, 0.0, -3.0, -2.0, 6.0,
              0.0, -3.0, -2.0, 6.0, 0.0, -3.0,
            ]
          );
        });
      });
    }

    describe('softmax', () => {
      it('forward, backward', async () => {
        const x = new Variable(
          ctor.fromArray([0, 1, 2, 3, -1, 3, 2, 4], [2, 4])
        );
        const y = await softmax(x);
        arrayNearlyEqual(
          await ta(y.data),
          [0.0321, 0.0871, 0.2369, 0.6439, 0.0045, 0.2436, 0.0896, 0.6623]
        );
        if (backend === 'cpu') {
          // backward of webgl is not yet implemented
          const weight = new Variable(
            ctor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])
          );
          const yr = await mul(y, weight);
          const s = await sum(yr);
          await s.backward();
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              -0.0799, -0.1301, -0.1167, 0.3267, -0.0108, -0.3435, -0.0367,
              0.3909,
            ]
          );
        }
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

    describe('flatten', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray(arange(24), [2, 3, 4]));
        const xr = await flatten(x);
        assert.deepEqual(xr.data.shape, [2, 12]);
        const weight = new Variable(ctor.fromArray(arange(100, 124), [2, 12]));
        const y = await mul(xr, weight);
        const s = await sum(y);
        await s.backward();
        assert.deepEqual(await ta(x.grad!.data), arange(100, 124));
      });
    });
  });
}

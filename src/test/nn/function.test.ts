/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import {
  add,
  bmm,
  cat,
  flatten,
  linear,
  matmul,
  mseLoss,
  mul,
  relu,
  reshape,
  sigmoid,
  softmax,
  softmaxCrossEntropy,
  split,
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

    describe('cat', () => {
      it('forward, backward', async () => {
        const x1 = new Variable(ctor.fromArray(arange(2 * 3), [2, 3]));
        const x2 = new Variable(ctor.fromArray(arange(100, 100 + 2 * 4), [2, 4]));
        const x3 = new Variable(ctor.fromArray(arange(200, 200 + 2 * 7), [2, 7]));
        const xr = await cat([x1, x2, x3], 1);
        assert.deepEqual(xr.data.shape, [2, 14]);
        assert.deepEqual(
          await ta(xr.data),
          [0, 1, 2, 100, 101, 102, 103, 200, 201, 202, 203, 204, 205, 206, 3, 4, 5, 104, 105, 106, 107, 207, 208, 209, 210, 211, 212, 213]
        );
        const weight = new Variable(ctor.fromArray(arange(100, 128), [2, 14]));
        const y = await mul(xr, weight);
        const s = await sum(y);
        await s.backward();
        assert.deepEqual(await ta(x1.grad!.data), [100, 101, 102, 114, 115, 116]);
        assert.deepEqual(await ta(x2.grad!.data), [103, 104, 105, 106, 117, 118, 119, 120]);
        assert.deepEqual(await ta(x3.grad!.data), [107, 108, 109, 110, 111, 112, 113, 121, 122, 123, 124, 125, 126, 127]);
      });
    });

    describe('split', () => {
      it('forward, backward', async () => {
        const x = new Variable(ctor.fromArray(arange(100, 100 + 3 * 4 * 4), [3, 4, 4]));
        const xr = await split(x, [1, 2, 1], 2);
        assert.deepEqual(xr[0].data.shape, [3, 4, 1]);
        assert.deepEqual(xr[1].data.shape, [3, 4, 2]);
        assert.deepEqual(xr[2].data.shape, [3, 4, 1]);
        assert.deepEqual(
          await ta(xr[0].data),
          [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144]
        );
        assert.deepEqual(
          await ta(xr[1].data),
          [101, 102, 105, 106, 109, 110, 113, 114, 117, 118, 121, 122, 125, 126,
            129, 130, 133, 134, 137, 138, 141, 142, 145, 146]
        );
        assert.deepEqual(
          await ta(xr[2].data),
          [103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147]
        );

        const w0 = new Variable(ctor.fromArray(arange(10, 22), [3, 4, 1]));
        const w1 = new Variable(ctor.fromArray(arange(20, 44), [3, 4, 2]));
        const w2 = new Variable(ctor.fromArray(arange(30, 42), [3, 4, 1]));
        const y0 = await sum(await mul(xr[0], w0));
        const y1 = await sum(await mul(xr[1], w1));
        const y2 = await sum(await mul(xr[2], w2));
        const s = await add(await add(y0, y1), y2);
        await s.backward();

        assert.deepEqual(await ta(x.grad!.data), [10, 20, 21, 30, 11, 22, 23, 31, 12, 24, 25, 32, 13, 26, 27, 33, 14, 28,
          29, 34, 15, 30, 31, 35, 16, 32, 33, 36, 17, 34, 35, 37, 18, 36, 37, 38,
          19, 38, 39, 39, 20, 40, 41, 40, 21, 42, 43, 41]);
      });
    });
  });
}

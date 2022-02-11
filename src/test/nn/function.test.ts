/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import {
  flatten,
  linear,
  matmul,
  max_pool2d,
  max_pool2d_with_indices,
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

    describe('maxpool', () => {
      if (backend === 'webgpu') {
        // not implemented
        return;
      }
      it('forward', async () => {
        let y: Variable, t: Variable;
        const x = new Variable(
          ctor.fromArray(
            [
              49.0, -43.0, -13.0, 37.0, -45.0, -94.0, 5.0, 26.0, 33.0, 3.0,
              -27.0, -19.0, -8.0, -85.0, 92.0, 98.0, 49.0, 41.0, 17.0, 60.0,
              30.0, 54.0, 98.0, -18.0, 88.0, -82.0, 90.0, -51.0, 3.0, -82.0,
              71.0, 44.0, 32.0, -74.0, -78.0, 26.0, 85.0, 55.0, 66.0, 64.0,
              -46.0, -92.0, 36.0, 46.0, 65.0, -7.0, -50.0, -17.0, -30.0, 6.0,
              -4.0, 57.0, -68.0, -66.0, -35.0, -8.0, -74.0, -16.0, -78.0, 27.0,
              37.0, -9.0, -44.0, 95.0, 54.0, -47.0, 86.0, -87.0, 4.0, -33.0,
              50.0, -59.0, -80.0, 87.0, -19.0, -83.0, 98.0, -26.0, -61.0, -51.0,
              -48.0, 15.0, -4.0, 97.0, 69.0, -93.0, -59.0, 80.0, -21.0, -63.0,
              45.0, 82.0, -43.0, 67.0, -80.0, -12.0, -42.0, -7.0, 56.0, -74.0,
              20.0, -27.0, 83.0, -6.0, 10.0, 13.0, 16.0, -18.0, 82.0, -74.0,
              78.0, 86.0, 58.0, -21.0, -26.0, 48.0, 88.0, 38.0, -16.0, 77.0,
              76.0, 97.0, -87.0, 68.0, 31.0, -93.0, -82.0, -74.0, 81.0, 66.0,
              -68.0, 83.0, 69.0, -28.0, -3.0, 33.0, 26.0, 12.0, 58.0, -69.0,
              50.0, -42.0, 52.0, 6.0, 32.0, -39.0, 49.0, 26.0, -53.0, 88.0,
              -47.0, -1.0, 94.0, 47.0, -13.0, -60.0, 91.0, -83.0, 21.0, -24.0,
              -18.0, -42.0, 22.0, 73.0, -1.0, 89.0, 59.0, -75.0, 9.0, 13.0,
              -79.0, 92.0, -92.0, 21.0, -4.0, 33.0, -29.0, -91.0, 46.0, -19.0,
              -62.0, -28.0, -9.0, 41.0, -32.0, 58.0, 99.0, 5.0, -24.0, -96.0,
              -11.0, 84.0, -71.0, -96.0, -51.0, 53.0, 18.0, 53.0, -16.0, -99.0,
              -85.0, 50.0, 10.0, 92.0, 83.0, 58.0, -91.0, -87.0, 77.0, 74.0,
              75.0, -58.0, -94.0, 51.0, 72.0, -81.0, -76.0, 20.0, 80.0, 15.0,
              -26.0, -9.0, 67.0, 67.0,
            ],
            [2, 2, 7, 8]
          )
        );

        y = await max_pool2d(x, { kernelSize: 3 });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 60.0, 90.0, 85.0, 87.0, 98.0, 56.0, 97.0, 97.0, 88.0, 94.0,
            91.0, 99.0, 92.0, 77.0, 92.0,
          ]
        );

        y = await max_pool2d(x, { kernelSize: 4 });
        assert.deepEqual(
          await ta(y.data),
          [90.0, 98.0, 97.0, 98.0, 97.0, 88.0, 99.0, 84.0]
        );

        y = await max_pool2d(x, { kernelSize: 3, padding: 1 });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 37.0, 98.0, 88.0, 90.0, 98.0, 6.0, 65.0, -7.0, 54.0, 86.0,
            95.0, 87.0, 98.0, 80.0, 13.0, 82.0, 86.0, 97.0, 88.0, 77.0, 81.0,
            83.0, 88.0, 94.0, 91.0, 89.0, 13.0, 92.0, 41.0, 58.0, 99.0, 84.0,
            77.0, 80.0, 72.0,
          ]
        );

        y = await max_pool2d(x, { kernelSize: 3, padding: 1, dilation: 2 });
        assert.deepEqual(
          await ta(y.data),
          [
            3.0, 92.0, 60.0, 98.0, 97.0, 86.0, 87.0, 98.0, 97.0, 58.0, 83.0,
            69.0, 53.0, 46.0, 92.0, 99.0,
          ]
        );

        y = await max_pool2d(x, { kernelSize: 3, stride: 2 });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 60.0, 98.0, 90.0, 90.0, 98.0, 36.0, 85.0, 85.0, 87.0, 98.0,
            98.0, 87.0, 98.0, 98.0, 56.0, 82.0, 83.0, 97.0, 88.0, 88.0, 81.0,
            83.0, 88.0, 94.0, 91.0, 91.0, 99.0, 99.0, 21.0, 99.0, 99.0, 83.0,
            80.0, 92.0, 83.0,
          ]
        );

        y = await max_pool2d(x, { kernelSize: 3, stride: 2, ceilMode: true });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 60.0, 98.0, 98.0, 90.0, 90.0, 98.0, 98.0, 36.0, 85.0, 85.0,
            66.0, 87.0, 98.0, 98.0, 95.0, 87.0, 98.0, 98.0, 80.0, 56.0, 82.0,
            83.0, 86.0, 97.0, 88.0, 88.0, 77.0, 81.0, 83.0, 88.0, 52.0, 94.0,
            91.0, 91.0, 59.0, 99.0, 99.0, 21.0, 84.0, 99.0, 99.0, 83.0, 84.0,
            80.0, 92.0, 83.0, 72.0,
          ]
        );

        [y, t] = await max_pool2d_with_indices(x, {
          kernelSize: 3,
          returnIndices: true,
        });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 60.0, 90.0, 85.0, 87.0, 98.0, 56.0, 97.0, 97.0, 88.0, 94.0,
            91.0, 99.0, 92.0, 77.0, 92.0,
          ]
        );
        assert.deepEqual(
          await ta(t.data),
          [0, 19, 26, 36, 17, 20, 42, 27, 9, 4, 40, 44, 18, 3, 40, 35]
        );

        [y, t] = await max_pool2d_with_indices(x, {
          kernelSize: 3,
          padding: 1,
          returnIndices: 'spatial',
        });
        assert.deepEqual(
          await ta(y.data),
          [
            49.0, 37.0, 98.0, 88.0, 90.0, 98.0, 6.0, 65.0, -7.0, 54.0, 86.0,
            95.0, 87.0, 98.0, 80.0, 13.0, 82.0, 86.0, 97.0, 88.0, 77.0, 81.0,
            83.0, 88.0, 94.0, 91.0, 89.0, 13.0, 92.0, 41.0, 58.0, 99.0, 84.0,
            77.0, 80.0, 72.0,
          ]
        );
        assert.deepEqual(
          await ta(t.data),
          [
            0, 3, 15, 24, 26, 22, 49, 44, 45, 8, 10, 7, 17, 20, 31, 49, 52, 55,
            9, 4, 7, 16, 19, 37, 40, 44, 53, 1, 3, 15, 17, 18, 23, 40, 50, 46,
          ]
        );
      });

      it('backward basic', async () => {
        const x = new Variable(
          ctor.fromArray(
            [
              49.0, -43.0, -13.0, 37.0, -45.0, -94.0, 5.0, 26.0, 33.0, 3.0,
              -27.0, -19.0, -8.0, -85.0, 92.0, 98.0, 49.0, 41.0, 17.0, 60.0,
              30.0, 54.0, 98.0, -18.0, 88.0, -82.0, 90.0, -51.0, 3.0, -82.0,
              71.0, 44.0, 32.0, -74.0, -78.0, 26.0, 85.0, 55.0, 66.0, 64.0,
              -46.0, -92.0, 36.0, 46.0, 65.0, -7.0, -50.0, -17.0, -30.0, 6.0,
              -4.0, 57.0, -68.0, -66.0, -35.0, -8.0, -74.0, -16.0, -78.0, 27.0,
              37.0, -9.0, -44.0, 95.0, 54.0, -47.0, 86.0, -87.0, 4.0, -33.0,
              50.0, -59.0, -80.0, 87.0, -19.0, -83.0, 98.0, -26.0, -61.0, -51.0,
              -48.0, 15.0, -4.0, 97.0, 69.0, -93.0, -59.0, 80.0, -21.0, -63.0,
              45.0, 82.0, -43.0, 67.0, -80.0, -12.0, -42.0, -7.0, 56.0, -74.0,
              20.0, -27.0, 83.0, -6.0, 10.0, 13.0, 16.0, -18.0, 82.0, -74.0,
              78.0, 86.0, 58.0, -21.0, -26.0, 48.0, 88.0, 38.0, -16.0, 77.0,
              76.0, 97.0, -87.0, 68.0, 31.0, -93.0, -82.0, -74.0, 81.0, 66.0,
              -68.0, 83.0, 69.0, -28.0, -3.0, 33.0, 26.0, 12.0, 58.0, -69.0,
              50.0, -42.0, 52.0, 6.0, 32.0, -39.0, 49.0, 26.0, -53.0, 88.0,
              -47.0, -1.0, 94.0, 47.0, -13.0, -60.0, 91.0, -83.0, 21.0, -24.0,
              -18.0, -42.0, 22.0, 73.0, -1.0, 89.0, 59.0, -75.0, 9.0, 13.0,
              -79.0, 92.0, -92.0, 21.0, -4.0, 33.0, -29.0, -91.0, 46.0, -19.0,
              -62.0, -28.0, -9.0, 41.0, -32.0, 58.0, 99.0, 5.0, -24.0, -96.0,
              -11.0, 84.0, -71.0, -96.0, -51.0, 53.0, 18.0, 53.0, -16.0, -99.0,
              -85.0, 50.0, 10.0, 92.0, 83.0, 58.0, -91.0, -87.0, 77.0, 74.0,
              75.0, -58.0, -94.0, 51.0, 72.0, -81.0, -76.0, 20.0, 80.0, 15.0,
              -26.0, -9.0, 67.0, 67.0,
            ],
            [2, 2, 7, 8]
          )
        );

        const y = await max_pool2d(x, { kernelSize: 3 });
        const weight = new Variable(
          ctor.fromArray(arange(y.data.size), y.data.shape)
        );
        const z = await mul(y, weight);
        const w = await sum(z);

        assert.deepEqual(await ta(w.data), [10699]);
        await w.backward();

        assert.deepEqual(
          await ta(x.grad!.data),
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0,
            0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0,
            14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
          ]
        );
      });

      it('backward pad', async () => {
        const x = new Variable(
          ctor.fromArray(
            [
              49.0, -43.0, -13.0, 37.0, -45.0, -94.0, 5.0, 26.0, 33.0, 3.0,
              -27.0, -19.0, -8.0, -85.0, 92.0, 98.0, 49.0, 41.0, 17.0, 60.0,
              30.0, 54.0, 98.0, -18.0, 88.0, -82.0, 90.0, -51.0, 3.0, -82.0,
              71.0, 44.0, 32.0, -74.0, -78.0, 26.0, 85.0, 55.0, 66.0, 64.0,
              -46.0, -92.0, 36.0, 46.0, 65.0, -7.0, -50.0, -17.0, -30.0, 6.0,
              -4.0, 57.0, -68.0, -66.0, -35.0, -8.0, -74.0, -16.0, -78.0, 27.0,
              37.0, -9.0, -44.0, 95.0, 54.0, -47.0, 86.0, -87.0, 4.0, -33.0,
              50.0, -59.0, -80.0, 87.0, -19.0, -83.0, 98.0, -26.0, -61.0, -51.0,
              -48.0, 15.0, -4.0, 97.0, 69.0, -93.0, -59.0, 80.0, -21.0, -63.0,
              45.0, 82.0, -43.0, 67.0, -80.0, -12.0, -42.0, -7.0, 56.0, -74.0,
              20.0, -27.0, 83.0, -6.0, 10.0, 13.0, 16.0, -18.0, 82.0, -74.0,
              78.0, 86.0, 58.0, -21.0, -26.0, 48.0, 88.0, 38.0, -16.0, 77.0,
              76.0, 97.0, -87.0, 68.0, 31.0, -93.0, -82.0, -74.0, 81.0, 66.0,
              -68.0, 83.0, 69.0, -28.0, -3.0, 33.0, 26.0, 12.0, 58.0, -69.0,
              50.0, -42.0, 52.0, 6.0, 32.0, -39.0, 49.0, 26.0, -53.0, 88.0,
              -47.0, -1.0, 94.0, 47.0, -13.0, -60.0, 91.0, -83.0, 21.0, -24.0,
              -18.0, -42.0, 22.0, 73.0, -1.0, 89.0, 59.0, -75.0, 9.0, 13.0,
              -79.0, 92.0, -92.0, 21.0, -4.0, 33.0, -29.0, -91.0, 46.0, -19.0,
              -62.0, -28.0, -9.0, 41.0, -32.0, 58.0, 99.0, 5.0, -24.0, -96.0,
              -11.0, 84.0, -71.0, -96.0, -51.0, 53.0, 18.0, 53.0, -16.0, -99.0,
              -85.0, 50.0, 10.0, 92.0, 83.0, 58.0, -91.0, -87.0, 77.0, 74.0,
              75.0, -58.0, -94.0, 51.0, 72.0, -81.0, -76.0, 20.0, 80.0, 15.0,
              -26.0, -9.0, 67.0, 67.0,
            ],
            [2, 2, 7, 8]
          )
        );

        const y = await max_pool2d(x, { kernelSize: 3, padding: 1 });
        const weight = new Variable(
          ctor.fromArray(arange(y.data.size), y.data.shape)
        );
        const z = await mul(y, weight);
        const w = await sum(z);

        assert.deepEqual(await ta(w.data), [47328]);
        await w.backward();

        assert.deepEqual(
          await ta(x.grad!.data),
          [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 3.0, 0.0,
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 9.0,
            0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 13.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 15.0, 0.0, 0.0, 16.0, 0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 19.0,
            0.0, 0.0, 20.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21.0, 0.0,
            0.0, 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0,
            25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 0.0, 0.0, 0.0,
            27.0, 0.0, 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 29.0, 0.0, 30.0, 31.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0, 34.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
          ]
        );
      });
    });

    it('backward stride < kernelSize', async () => {
      // stride < kernelSize の場合、x.gradの同じ要素に複数の勾配が足し合わさる場合がある
      const x = new Variable(
        ctor.fromArray(
          [
            49.0, -43.0, -13.0, 37.0, -45.0, -94.0, 5.0, 26.0, 33.0, 3.0, -27.0,
            -19.0, -8.0, -85.0, 92.0, 98.0, 49.0, 41.0, 17.0, 60.0, 30.0, 54.0,
            98.0, -18.0, 88.0, -82.0, 90.0, -51.0, 3.0, -82.0, 71.0, 44.0, 32.0,
            -74.0, -78.0, 26.0, 85.0, 55.0, 66.0, 64.0, -46.0, -92.0, 36.0,
            46.0, 65.0, -7.0, -50.0, -17.0, -30.0, 6.0, -4.0, 57.0, -68.0,
            -66.0, -35.0, -8.0, -74.0, -16.0, -78.0, 27.0, 37.0, -9.0, -44.0,
            95.0, 54.0, -47.0, 86.0, -87.0, 4.0, -33.0, 50.0, -59.0, -80.0,
            87.0, -19.0, -83.0, 98.0, -26.0, -61.0, -51.0, -48.0, 15.0, -4.0,
            97.0, 69.0, -93.0, -59.0, 80.0, -21.0, -63.0, 45.0, 82.0, -43.0,
            67.0, -80.0, -12.0, -42.0, -7.0, 56.0, -74.0, 20.0, -27.0, 83.0,
            -6.0, 10.0, 13.0, 16.0, -18.0, 82.0, -74.0, 78.0, 86.0, 58.0, -21.0,
            -26.0, 48.0, 88.0, 38.0, -16.0, 77.0, 76.0, 97.0, -87.0, 68.0, 31.0,
            -93.0, -82.0, -74.0, 81.0, 66.0, -68.0, 83.0, 69.0, -28.0, -3.0,
            33.0, 26.0, 12.0, 58.0, -69.0, 50.0, -42.0, 52.0, 6.0, 32.0, -39.0,
            49.0, 26.0, -53.0, 88.0, -47.0, -1.0, 94.0, 47.0, -13.0, -60.0,
            91.0, -83.0, 21.0, -24.0, -18.0, -42.0, 22.0, 73.0, -1.0, 89.0,
            59.0, -75.0, 9.0, 13.0, -79.0, 92.0, -92.0, 21.0, -4.0, 33.0, -29.0,
            -91.0, 46.0, -19.0, -62.0, -28.0, -9.0, 41.0, -32.0, 58.0, 99.0,
            5.0, -24.0, -96.0, -11.0, 84.0, -71.0, -96.0, -51.0, 53.0, 18.0,
            53.0, -16.0, -99.0, -85.0, 50.0, 10.0, 92.0, 83.0, 58.0, -91.0,
            -87.0, 77.0, 74.0, 75.0, -58.0, -94.0, 51.0, 72.0, -81.0, -76.0,
            20.0, 80.0, 15.0, -26.0, -9.0, 67.0, 67.0,
          ],
          [2, 2, 7, 8]
        )
      );

      const y = await max_pool2d(x, { kernelSize: 3, stride: 2 });
      const weight = new Variable(
        ctor.fromArray(arange(y.data.size), y.data.shape)
      );
      const z = await mul(y, weight);
      const w = await sum(z);

      assert.deepEqual(await ta(w.data), [54009]);
      await w.backward();

      assert.deepEqual(
        await ta(x.grad!.data),
        [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 7.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 15.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 22.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0,
          0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 51.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 116.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.0,
          67.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          33.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
      );
    });
  });
}

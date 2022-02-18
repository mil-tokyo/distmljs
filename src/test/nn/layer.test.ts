/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Backend } from '../../backend';
import { Layer, Variable } from '../../nn/core';
import { relu, sum } from '../../nn/functions';
import { Conv2d, Linear } from '../../nn/layers';
import { Tensor, WebGPUTensor } from '../../tensor';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGLTensor } from '../../tensor/webgl/webglTensor';
import { arange } from '../../util';
import { testFlag } from '../testFlag';

class Model extends Layer {
  l1: Linear;
  l2: Linear;

  constructor(inFeatures: number, hidden: number, outFeatures: number) {
    super();
    this.l1 = new Linear(inFeatures, hidden);
    this.l2 = new Linear(hidden, outFeatures);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.l1.c(y);
    y = await relu(y);
    y = await this.l2.c(y);
    return [y];
  }
}

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

      it('training / eval mode', async () => {
        const model = new Model(3, 2, 4);
        // default is train
        assert.isTrue(model.training);
        assert.isTrue(model.l1.training);
        assert.isTrue(model.l2.training);
        model.eval();
        assert.isFalse(model.training);
        // should recursively apply
        assert.isFalse(model.l1.training);
        assert.isFalse(model.l2.training);
        model.train();
        assert.isTrue(model.training);
        assert.isTrue(model.l1.training);
        assert.isTrue(model.l2.training);
      });
    });

    describe('conv2d', () => {
      it('forward / backward', async () => {
        const conv2d = new Conv2d(3, 2, 3, { padding: 1, stride: 2 });
        assert.deepEqual(conv2d.weight.data.shape, [2, 3, 3, 3]);
        assert.deepEqual(conv2d.bias!.data.shape, [2]);
        (conv2d.weight.data as CPUTensor).setArray([
          -2.0, 4.0, 1.0, 1.0, 3.0, 1.0, -4.0, -5.0, 3.0, -1.0, 0.0, -1.0, 3.0,
          3.0, -4.0, 3.0, 1.0, 5.0, 3.0, 0.0, 0.0, 3.0, -2.0, 5.0, -3.0, 3.0,
          -3.0, -1.0, -1.0, -2.0, 1.0, 1.0, 4.0, -4.0, 1.0, 1.0, 2.0, -1.0, 0.0,
          -1.0, 4.0, -5.0, -1.0, 2.0, -5.0, 0.0, -1.0, -3.0, -3.0, 1.0, -4.0,
          -2.0, 5.0, -4.0,
        ]);
        (conv2d.bias!.data as CPUTensor).setArray([10, 20]);
        await conv2d.to(backend);

        const x = new Variable(
          ctor.fromArray(
            [
              5.0, 2.0, -4.0, -2.0, 3.0, -2.0, -1.0, -5.0, 1.0, 3.0, 4.0, -1.0,
              -3.0, 0.0, 3.0, 1.0, 0.0, 5.0, 5.0, 5.0, -3.0, -2.0, 3.0, -2.0,
              -4.0, -4.0, 4.0, 5.0, 4.0, 4.0, -5.0, -1.0, 3.0, -2.0, 4.0, -5.0,
              -2.0, 3.0, -4.0, -2.0, 3.0, 5.0, 4.0, 2.0, 4.0, 2.0, 5.0, -1.0,
              3.0, 5.0, -2.0, 2.0, 3.0, 5.0, 2.0, 4.0, -2.0, 0.0, 5.0, 3.0, 0.0,
              5.0, -2.0, -2.0, 5.0, -2.0, 5.0, -1.0, -3.0, 3.0, 0.0, 5.0, -2.0,
              1.0, 4.0, -3.0, 5.0, -1.0, -1.0, 5.0, 2.0, 2.0, -4.0, -3.0, -3.0,
              -1.0, -4.0, 3.0, 4.0, 1.0, -2.0, 2.0, -5.0, -5.0, 5.0, -2.0, -4.0,
              1.0, -1.0, -4.0, 3.0, 4.0, 1.0, -1.0, 3.0, -5.0, -2.0, 0.0, 4.0,
              -4.0, -2.0, -1.0, -2.0, -4.0, -5.0, 3.0, 0.0, -2.0, -3.0, -2.0,
              -5.0, 3.0, 2.0, -3.0, -3.0, -5.0, 5.0, -3.0, -2.0, -4.0, 3.0,
              -1.0, 5.0, 5.0, -5.0, 0.0, -1.0, -3.0, 4.0, -4.0, 1.0, 2.0, -2.0,
              -2.0, 3.0, 0.0, 1.0, 0.0, 3.0, -4.0,
            ],
            [2, 3, 5, 5]
          )
        );
        const y = await conv2d.c(x);
        assert.deepEqual(
          await ta(y.data),
          [
            28.0, 47.0, 25.0, 33.0, 18.0, -5.0, 9.0, 72.0, 14.0, 7.0, -16.0,
            34.0, -47.0, 25.0, 27.0, -45.0, -31.0, 19.0, 27.0, 61.0, 37.0, 24.0,
            0.0, 14.0, -28.0, 33.0, 23.0, 27.0, -11.0, 1.0, 6.0, 3.0, 59.0,
            -49.0, 41.0, -14.0,
          ]
        );
        const z = await sum(y);
        await z.backward();
        assert.deepEqual(
          await ta(conv2d.weight.grad!.data),
          [
            1.0, 0.0, 1.0, -6.0, 3.0, -6.0, 1.0, 0.0, 1.0, 3.0, 3.0, 3.0, 8.0,
            -1.0, 8.0, 3.0, 3.0, 3.0, 9.0, 12.0, 9.0, 26.0, -15.0, 26.0, 9.0,
            12.0, 9.0, 1.0, 0.0, 1.0, -6.0, 3.0, -6.0, 1.0, 0.0, 1.0, 3.0, 3.0,
            3.0, 8.0, -1.0, 8.0, 3.0, 3.0, 3.0, 9.0, 12.0, 9.0, 26.0, -15.0,
            26.0, 9.0, 12.0, 9.0,
          ]
        );
        assert.deepEqual(await ta(conv2d.bias!.grad!.data), [18.0, 18.0]);
        assert.deepEqual(
          await ta(x.grad!.data),
          [
            4.0, 7.0, 4.0, 7.0, 4.0, -1.0, -8.0, -1.0, -8.0, -1.0, 4.0, 7.0,
            4.0, 7.0, 4.0, -1.0, -8.0, -1.0, -8.0, -1.0, 4.0, 7.0, 4.0, 7.0,
            4.0, 7.0, -7.0, 7.0, -7.0, 7.0, 2.0, 2.0, 2.0, 2.0, 2.0, 7.0, -7.0,
            7.0, -7.0, 7.0, 2.0, 2.0, 2.0, 2.0, 2.0, 7.0, -7.0, 7.0, -7.0, 7.0,
            -1.0, 1.0, -1.0, 1.0, -1.0, 7.0, -12.0, 7.0, -12.0, 7.0, -1.0, 1.0,
            -1.0, 1.0, -1.0, 7.0, -12.0, 7.0, -12.0, 7.0, -1.0, 1.0, -1.0, 1.0,
            -1.0, 4.0, 7.0, 4.0, 7.0, 4.0, -1.0, -8.0, -1.0, -8.0, -1.0, 4.0,
            7.0, 4.0, 7.0, 4.0, -1.0, -8.0, -1.0, -8.0, -1.0, 4.0, 7.0, 4.0,
            7.0, 4.0, 7.0, -7.0, 7.0, -7.0, 7.0, 2.0, 2.0, 2.0, 2.0, 2.0, 7.0,
            -7.0, 7.0, -7.0, 7.0, 2.0, 2.0, 2.0, 2.0, 2.0, 7.0, -7.0, 7.0, -7.0,
            7.0, -1.0, 1.0, -1.0, 1.0, -1.0, 7.0, -12.0, 7.0, -12.0, 7.0, -1.0,
            1.0, -1.0, 1.0, -1.0, 7.0, -12.0, 7.0, -12.0, 7.0, -1.0, 1.0, -1.0,
            1.0, -1.0,
          ]
        );
      });
    });
  });
}

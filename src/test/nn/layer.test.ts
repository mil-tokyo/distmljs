/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Backend } from '../../backend';
import { Layer, Variable } from '../../nn/core';
import { mseLoss, relu, sum } from '../../nn/functions';
import { BatchNorm, Conv2d, Linear } from '../../nn/layers';
import { SGD } from '../../nn/optimizers';
import { Tensor, WebGPUTensor } from '../../tensor';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGLTensor } from '../../tensor/webgl/webglTensor';
import { arange } from '../../util';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

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

    describe('batchNorm', () => {
      it('forward / backward', async () => {
        const bn = new BatchNorm(3, {});
        (bn.weight!.data as CPUTensor).setArray([5.0, 2.0, -4.0]);
        (bn.bias!.data as CPUTensor).setArray([-2.0, 3.0, -2.0]);
        bn.train();
        await bn.to(backend);
        const opt = new SGD(bn.parameters(), 0.1, 0.0);

        // iter 1
        {
          opt.zeroGrad();
          const x = new Variable(
            ctor.fromArray(
              [
                -1.0, -5.0, 1.0, 3.0, 4.0, -1.0, -3.0, 0.0, 3.0, 1.0, 0.0, 5.0,
                5.0, 5.0, -3.0, -2.0, 3.0, -2.0, -4.0, -4.0, 4.0, 5.0, 4.0, 4.0,
              ],
              [2, 3, 2, 2]
            )
          );
          const y = await bn.c(x);
          arrayNearlyEqual(
            await ta(y.data),
            [
              -3.96553897857666, -9.683469772338867, -1.106573224067688,
              1.752392292022705, 6.4238176345825195, 2.912209987640381,
              1.5075666904449463, 3.6145315170288086, -1.4165410995483398,
              3.2511308193206787, 5.584966659545898, -6.084212303161621,
              4.611358165740967, 4.611358165740967, -6.824504375457764,
              -5.395021438598633, 5.72149658203125, 2.209888458251953,
              0.8052451610565186, 0.8052451610565186, -3.7503767013549805,
              -6.084212303161621, -3.7503767013549805, -3.7503767013549805,
            ]
          );
          const target = new Variable(
            ctor.fromArray(
              [
                0.0, -1.0, 1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
              ],
              [2, 3, 2, 2]
            )
          );
          const loss = await mseLoss(y, target);
          arrayNearlyEqual(await ta(loss.data), [20.887048721313477]);
          await loss.backward();
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              -0.001673722406849265, 0.11258099228143692, -0.11836281418800354,
              0.12231877446174622, -0.0020295772701501846, 0.06608228385448456,
              0.011389481835067272, -0.05288832634687424, 0.1158638671040535,
              -0.12414366751909256, 0.04758226126432419, -0.22758780419826508,
              0.0056296722032129765, 0.0056296722032129765,
              -0.004108187276870012, -0.12201446294784546,
              -0.00011253452248638496, 0.009472491219639778,
              -0.045220304280519485, 0.013306503184139729, -0.0558619387447834,
              0.16138489544391632, 0.13862434029579163, -0.0558619387447834,
            ]
          );
          arrayNearlyEqual(
            await ta(bn.weight!.grad!.data),
            [3.3095061779022217, 1.395516276359558, -2.5329480171203613]
          );
          arrayNearlyEqual(
            await ta(bn.bias!.grad!.data),
            [-1.3333332538604736, 1.9166667461395264, -1.5833332538604736]
          );
          arrayNearlyEqual(
            await ta(bn.runningMean!.data),
            [0.03750000149011612, -0.08749999850988388, 0.32499998807907104]
          );
          arrayNearlyEqual(
            await ta(bn.runningVar!.data),
            [2.2982141971588135, 1.8267858028411865, 1.235714316368103]
          );
          arrayNearlyEqual(await ta(bn.numBatchesTracked!.data), [1]);
          await opt.step();
        }

        // iter 2
        {
          opt.zeroGrad();
          const x = new Variable(
            ctor.fromArray(
              [
                2.0, 4.0, -2.0, 0.0, 5.0, 3.0, 0.0, 5.0, -2.0, -2.0, 5.0, -2.0,
                5.0, -1.0, -3.0, 3.0, 0.0, 5.0, -2.0, 1.0, 4.0, -3.0, 5.0, -1.0,
              ],
              [2, 3, 2, 2]
            )
          );

          const y = await bn.c(x);
          arrayNearlyEqual(
            await ta(y.data),
            [
              -0.1617722511291504, 3.248016834259033, -6.981350421905518,
              -3.571561336517334, 4.88886833190918, 3.4415395259857178,
              1.2705469131469727, 4.88886833190918, 1.0151678323745728,
              1.0151678323745728, -6.983968734741211, 1.0151678323745728,
              4.952911853790283, -5.276455879211426, -8.68624496459961,
              1.5431222915649414, 1.2705469131469727, 4.88886833190918,
              -0.1767815351486206, 1.994211196899414, -5.841235160827637,
              2.1579017639160156, -6.983968734741211, -0.12756597995758057,
            ]
          );

          const target = new Variable(
            ctor.fromArray(
              [
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0,
              ],
              [2, 3, 2, 2]
            )
          );
          const loss = await mseLoss(y, target);
          arrayNearlyEqual(await ta(loss.data), [18.944927215576172]);

          await loss.backward();
          arrayNearlyEqual(
            await ta(x.grad!.data),
            [
              0.05091036856174469, -0.09589926898479462, -0.08169396221637726,
              -0.08642902970314026, -0.00997947808355093, 0.025804445147514343,
              0.04932768642902374, -0.00997947808355093, -0.004429475404322147,
              -0.004429475404322147, -0.058686431497335434, 0.09079834073781967,
              0.04380771890282631, 0.05801307410001755, 0.06274819374084473,
              0.04854283481836319, -0.010977668687701225, -0.00997947808355093,
              -0.035499077290296555, 0.0012830484192818403, 0.04429240524768829,
              0.0033214937429875135, 0.03654143586754799, -0.1074083000421524,
            ]
          );
          arrayNearlyEqual(
            await ta(bn.weight!.grad!.data),
            [3.1431243419647217, 0.8918421864509583, -2.675715923309326]
          );
          arrayNearlyEqual(
            await ta(bn.bias!.grad!.data),
            [-1.494444489479065, 1.7055556774139404, -1.394444465637207]
          );
          arrayNearlyEqual(
            await ta(bn.runningMean!.data),
            [0.13375000655651093, 0.13375000655651093, 0.3425000011920929]
          );
          arrayNearlyEqual(
            await ta(bn.runningVar!.data),
            [2.9255356788635254, 2.3994643688201904, 2.3407142162323]
          );
          arrayNearlyEqual(await ta(bn.numBatchesTracked!.data), [2]);
          await opt.step();
        }

        arrayNearlyEqual(
          await ta(bn.weight!.data),
          [4.354736804962158, 1.7712641954421997, -3.479133605957031]
        );
        arrayNearlyEqual(
          await ta(bn.bias!.data),
          [-1.7172222137451172, 2.637777805328369, -1.702222228050232]
        );
        // eval mode
        bn.eval();
        {
          const x = new Variable(
            ctor.fromArray(
              [
                1.0, -1.0, 3.0, -5.0, -2.0, 0.0, 4.0, -4.0, -2.0, -1.0, -2.0,
                -4.0, -5.0, 3.0, 0.0, -2.0, -3.0, -2.0, -5.0, 3.0, 2.0, -3.0,
                -3.0, -5.0,
              ],
              [2, 3, 2, 2]
            )
          );

          const y = await bn.c(x);
          arrayNearlyEqual(
            await ta(y.data),
            [
              0.488250732421875, -4.603750228881836, 5.580251693725586,
              -14.787752151489258, 0.19789576530456543, 2.4848384857177734,
              7.0587239265441895, -2.0890469551086426, 3.6246907711029053,
              1.3506617546081543, 3.6246907711029053, 8.172748565673828,
              -14.787752151489258, 5.580251693725586, -2.0577497482299805,
              -7.149750709533691, -0.9455757141113281, 0.19789576530456543,
              -3.232518196105957, 5.915252685546875, -5.4714250564575195,
              5.898719787597656, 5.898719787597656, 10.44677734375,
            ]
          );
        }
      });
    });
  });
}

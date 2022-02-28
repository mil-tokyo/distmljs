/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { assert } from 'chai';
import { Variable } from '../../nn/core';
import { conv2d, Conv2dParams, mul, sum } from '../../nn/functions';
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
  describe(`nn/function/conv/${backend}`, () => {
    describe('conv2d', () => {
      if (backend === 'webgpu') {
        // not implemented
        return;
      }

      const doConv2d = async (
        params: Conv2dParams,
        xData: { data: number[]; shape: number[] },
        weightData: { data: number[]; shape: number[] },
        biasData: { data: number[]; shape: number[] } | null,
        expectedYShape: number[],
        expectedY: number[],
        expectedXGrad?: number[],
        expectedWeightGrad?: number[],
        expectedBiasGrad?: number[]
      ) => {
        const x = new Variable(ctor.fromArray(xData.data, xData.shape));
        const weight = new Variable(
          ctor.fromArray(weightData.data, weightData.shape)
        );
        let y: Variable;
        let bias: Variable | undefined = undefined;
        if (biasData) {
          bias = new Variable(ctor.fromArray(biasData.data, biasData.shape));

          y = await conv2d(x, weight, bias, params);
        } else {
          y = await conv2d(x, weight, params);
        }
        assert.deepEqual(y.data.shape, expectedYShape, 'y.shape');
        arrayNearlyEqual(await ta(y.data), expectedY, 'y');
        if (expectedXGrad || expectedWeightGrad || expectedBiasGrad) {
          const coef = new Variable(
            ctor.fromArray(arange(y.data.size), y.data.shape)
          );
          const z = await mul(y, coef);
          const w = await sum(z);
          await w.backward();
          if (expectedXGrad) {
            arrayNearlyEqual(await ta(x.grad!.data), expectedXGrad, 'gx');
          }
          if (expectedWeightGrad) {
            arrayNearlyEqual(
              await ta(weight.grad!.data),
              expectedWeightGrad,
              'gw'
            );
          }
          if (expectedBiasGrad && bias) {
            arrayNearlyEqual(await ta(bias.grad!.data), expectedBiasGrad, 'gb');
          }
        }
      };

      it('forward / backward stride', async () => {
        await doConv2d(
          { stride: 2 },
          {
            data: [
              6.0, -5.0, -9.0, -9.0, 2.0, -2.0, -4.0, 0.0, 0.0, -5.0, 4.0, -8.0,
              -6.0, -2.0, 7.0, 6.0, -8.0, 2.0, 9.0, -3.0, 1.0, 9.0, 0.0, 1.0,
              5.0, -8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0, -1.0, 9.0,
              -4.0, -9.0, -1.0, -6.0, -4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0,
              -6.0, 8.0, -1.0, 3.0, 4.0, -1.0, -9.0, 2.0, 3.0, 7.0, -4.0, 4.0,
              6.0, -7.0, 3.0, 9.0, 4.0, -4.0, 0.0, -6.0, 9.0, -8.0, -5.0, -5.0,
              -4.0, 2.0, -9.0, -3.0, 1.0, -7.0, -9.0, 5.0, 1.0, 3.0, 5.0, 1.0,
              6.0, 7.0, 1.0, 6.0, -8.0, 7.0, 7.0, 1.0, 2.0, -7.0, -7.0, -8.0,
              2.0, 1.0, -1.0, 4.0, -4.0, -1.0, -2.0, -7.0, -1.0, -4.0, 0.0, 7.0,
              -1.0, -4.0, 7.0, 7.0, 5.0, -4.0, -8.0, 1.0, 4.0, -6.0, 4.0, -8.0,
              5.0, 3.0, 4.0, 4.0, 5.0, -6.0, 1.0, 4.0, 4.0, 2.0, -2.0, 3.0,
              -5.0, -9.0, 5.0, -2.0, 9.0, 7.0, -8.0, 6.0, -9.0, -9.0, -9.0,
              -6.0, 5.0, -8.0, 9.0, -9.0, 1.0, 1.0, -2.0, 8.0, 3.0, -6.0, 7.0,
              -1.0, 3.0, 3.0, 8.0, -8.0, 6.0, -6.0, 3.0, 1.0, -9.0, 2.0, 9.0,
              2.0, -5.0, -7.0, -8.0, 9.0, 4.0, 7.0, 8.0, 5.0, 1.0, -4.0, 0.0,
              -5.0, -4.0, 9.0, -7.0, 6.0, 4.0, -6.0, -6.0, 7.0, -3.0, 3.0, 0.0,
              -8.0, -1.0, 5.0, -7.0, -4.0, -9.0, 4.0, -9.0, 8.0, 5.0, 7.0, -7.0,
              9.0, 2.0, -6.0, 0.0, -4.0, 4.0, 8.0, -4.0, -2.0, -1.0, 2.0, -4.0,
              6.0, 4.0, 4.0, 7.0, -3.0, 7.0, -5.0, 0.0, -6.0,
            ],
            shape: [2, 2, 7, 8],
          },
          {
            data: [
              -1.0, 1.0, 4.0, -5.0, 9.0, 9.0, -5.0, -4.0, 3.0, -2.0, 5.0, -7.0,
              4.0, -8.0, 4.0, 6.0, 1.0, 1.0, -6.0, 2.0, -7.0, 5.0, -6.0, 9.0,
              1.0, 1.0, 4.0, 1.0, 0.0, -6.0, 5.0, -6.0, -3.0, 2.0, 6.0, -4.0,
              7.0, -1.0, 3.0, 0.0, 1.0, -6.0, 5.0, -6.0, -7.0, 6.0, 9.0, 6.0,
              8.0, -8.0, 9.0, -4.0, 6.0, -6.0, 7.0, 7.0, 4.0, 7.0, 4.0, 9.0,
              6.0, -7.0, -8.0, 8.0, 5.0, 8.0, 1.0, -3.0, 9.0, -4.0, 4.0, 2.0,
            ],
            shape: [4, 2, 3, 3],
          },
          { data: [4.0, -1.0, -4.0, -8.0], shape: [4] },
          [2, 4, 3, 3],
          [
            81.0, -150.0, 20.0, -19.0, 124.0, 108.0, 88.0, -137.0, -17.0, 223.0,
            -26.0, 158.0, 92.0, -64.0, 121.0, -30.0, 32.0, -19.0, 75.0, 221.0,
            -3.0, -66.0, -13.0, -131.0, -80.0, 4.0, -162.0, -106.0, 44.0, -93.0,
            -6.0, 34.0, 61.0, -6.0, 20.0, -128.0, 46.0, 98.0, -144.0, -64.0,
            1.0, -189.0, 69.0, 14.0, 111.0, 126.0, 95.0, -74.0, 63.0, -41.0,
            -179.0, 51.0, -100.0, 88.0, -105.0, 77.0, 22.0, -32.0, 90.0, -177.0,
            12.0, 62.0, -329.0, -102.0, 93.0, 68.0, -72.0, -70.0, -394.0, 98.0,
            123.0, -142.0,
          ],
          [
            261.0, 189.0, 367.0, 198.0, 378.0, 207.0, 107.0, 0.0, 234.0, 72.0,
            457.0, 80.0, 485.0, 88.0, 258.0, 0.0, 543.0, -72.0, 362.0, -79.0,
            372.0, -86.0, -203.0, 0.0, 255.0, 96.0, 541.0, 104.0, 569.0, 112.0,
            321.0, 0.0, 585.0, -93.0, 392.0, -100.0, 402.0, -107.0, -215.0, 0.0,
            276.0, 120.0, 625.0, 128.0, 653.0, 136.0, 384.0, 0.0, 303.0, -384.0,
            -44.0, -400.0, -45.0, -416.0, -370.0, 0.0, 333.0, 297.0, 616.0,
            316.0, 630.0, 335.0, 272.0, 0.0, 216.0, -279.0, 612.0, -304.0,
            649.0, -329.0, 416.0, 0.0, 210.0, 624.0, 406.0, 660.0, 413.0, 696.0,
            171.0, 0.0, 270.0, -354.0, 723.0, -379.0, 760.0, -404.0, 473.0, 0.0,
            249.0, 732.0, 427.0, 768.0, 434.0, 804.0, 153.0, 0.0, 324.0, -429.0,
            834.0, -454.0, 871.0, -479.0, 530.0, 0.0, -162.0, 372.0, -294.0,
            389.0, -301.0, 406.0, -146.0, 0.0, 513.0, 513.0, 763.0, 522.0,
            774.0, 531.0, 251.0, 0.0, 486.0, 360.0, 1465.0, 368.0, 1493.0,
            376.0, 1014.0, 0.0, 1047.0, -324.0, 722.0, -331.0, 732.0, -338.0,
            -347.0, 0.0, 507.0, 384.0, 1549.0, 392.0, 1577.0, 400.0, 1077.0,
            0.0, 1089.0, -345.0, 752.0, -352.0, 762.0, -359.0, -359.0, 0.0,
            528.0, 408.0, 1633.0, 416.0, 1661.0, 424.0, 1140.0, 0.0, 555.0,
            -960.0, -80.0, -976.0, -81.0, -992.0, -658.0, 0.0, 801.0, 981.0,
            1120.0, 1000.0, 1134.0, 1019.0, 308.0, 0.0, 864.0, -1179.0, 1944.0,
            -1204.0, 1981.0, -1229.0, 1100.0, 0.0, 678.0, 1920.0, 658.0, 1956.0,
            665.0, 1992.0, -45.0, 0.0, 918.0, -1254.0, 2055.0, -1279.0, 2092.0,
            -1304.0, 1157.0, 0.0, 717.0, 2028.0, 679.0, 2064.0, 686.0, 2100.0,
            -63.0, 0.0, 972.0, -1329.0, 2166.0, -1354.0, 2203.0, -1379.0,
            1214.0, 0.0, -162.0, 984.0, -546.0, 1001.0, -553.0, 1018.0, -398.0,
            0.0,
          ],
          [
            527.0, -1138.0, 255.0, 736.0, 5.0, 190.0, 440.0, -829.0, 304.0,
            -552.0, 875.0, -454.0, -808.0, 602.0, -1810.0, 575.0, 329.0, 101.0,
            617.0, -1426.0, 246.0, 943.0, -121.0, 406.0, 503.0, -829.0, 367.0,
            -732.0, 1055.0, -634.0, -799.0, 692.0, -2134.0, 575.0, 347.0, 20.0,
            707.0, -1714.0, 237.0, 1150.0, -247.0, 622.0, 566.0, -829.0, 430.0,
            -912.0, 1235.0, -814.0, -790.0, 782.0, -2458.0, 575.0, 365.0, -61.0,
            797.0, -2002.0, 228.0, 1357.0, -373.0, 838.0, 629.0, -829.0, 493.0,
            -1092.0, 1415.0, -994.0, -781.0, 872.0, -2782.0, 575.0, 383.0,
            -142.0,
          ],
          [396.0, 558.0, 720.0, 882.0]
        );
        await doConv2d(
          {},
          {
            data: [
              6.0, -5.0, -9.0, -9.0, 2.0, -2.0, -4.0, 0.0, 0.0, -5.0, 4.0, -8.0,
              -6.0, -2.0, 7.0, 6.0, -8.0, 2.0, 9.0, -3.0, 1.0, 9.0, 0.0, 1.0,
              5.0, -8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0, -1.0, 9.0,
              -4.0, -9.0, -1.0, -6.0, -4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0,
              -6.0, 8.0, -1.0, 3.0, 4.0, -1.0, -9.0, 2.0, 3.0, 7.0, -4.0, 4.0,
              6.0, -7.0, 3.0, 9.0, 4.0, -4.0, 0.0, -6.0, 9.0, -8.0, -5.0, -5.0,
              -4.0, 2.0, -9.0, -3.0, 1.0, -7.0, -9.0, 5.0, 1.0, 3.0, 5.0, 1.0,
              6.0, 7.0, 1.0, 6.0, -8.0, 7.0, 7.0, 1.0, 2.0, -7.0, -7.0, -8.0,
              2.0, 1.0, -1.0, 4.0, -4.0, -1.0, -2.0, -7.0, -1.0, -4.0, 0.0, 7.0,
              -1.0, -4.0, 7.0, 7.0, 5.0, -4.0, -8.0, 1.0, 4.0, -6.0, 4.0, -8.0,
              5.0, 3.0, 4.0, 4.0, 5.0, -6.0, 1.0, 4.0, 4.0, 2.0, -2.0, 3.0,
              -5.0, -9.0, 5.0, -2.0, 9.0, 7.0, -8.0, 6.0, -9.0, -9.0, -9.0,
              -6.0, 5.0, -8.0, 9.0, -9.0, 1.0, 1.0, -2.0, 8.0, 3.0, -6.0, 7.0,
              -1.0, 3.0, 3.0, 8.0, -8.0, 6.0, -6.0, 3.0, 1.0, -9.0, 2.0, 9.0,
              2.0, -5.0, -7.0, -8.0, 9.0, 4.0, 7.0, 8.0, 5.0, 1.0, -4.0, 0.0,
              -5.0, -4.0, 9.0, -7.0, 6.0, 4.0, -6.0, -6.0, 7.0, -3.0, 3.0, 0.0,
              -8.0, -1.0, 5.0, -7.0, -4.0, -9.0, 4.0, -9.0, 8.0, 5.0, 7.0, -7.0,
              9.0, 2.0, -6.0, 0.0, -4.0, 4.0, 8.0, -4.0, -2.0, -1.0, 2.0, -4.0,
              6.0, 4.0, 4.0, 7.0, -3.0, 7.0, -5.0, 0.0, -6.0,
            ],
            shape: [2, 2, 7, 8],
          },
          {
            data: [
              -1.0, 1.0, 4.0, -5.0, 9.0, 9.0, -5.0, -4.0, 3.0, -2.0, 5.0, -7.0,
              4.0, -8.0, 4.0, 6.0, 1.0, 1.0, -6.0, 2.0, -7.0, 5.0, -6.0, 9.0,
              1.0, 1.0, 4.0, 1.0, 0.0, -6.0, 5.0, -6.0, -3.0, 2.0, 6.0, -4.0,
              7.0, -1.0, 3.0, 0.0, 1.0, -6.0, 5.0, -6.0, -7.0, 6.0, 9.0, 6.0,
              8.0, -8.0, 9.0, -4.0, 6.0, -6.0,
            ],
            shape: [3, 2, 3, 3],
          },
          { data: [7.0, 7.0, 4.0], shape: [3] },
          [2, 3, 5, 6],
          [
            84.0, -201.0, -147.0, -65.0, 23.0, -2.0, 85.0, 224.0, -170.0, 140.0,
            159.0, 126.0, -16.0, 109.0, 127.0, 51.0, 111.0, 71.0, -8.0, 58.0,
            -18.0, -198.0, 57.0, -43.0, 91.0, 124.0, -134.0, -66.0, -14.0,
            121.0, 231.0, -97.0, -18.0, 105.0, 166.0, -118.0, -2.0, 184.0, 16.0,
            180.0, 141.0, 95.0, 100.0, 136.0, -56.0, -38.0, 129.0, -141.0,
            -172.0, 119.0, -108.0, -90.0, 17.0, -81.0, -22.0, -101.0, 40.0,
            157.0, -11.0, 148.0, 83.0, -120.0, 229.0, -241.0, 5.0, -83.0,
            -122.0, -107.0, -142.0, -213.0, -90.0, -208.0, -58.0, -153.0, -5.0,
            168.0, -123.0, 94.0, -10.0, -106.0, 313.0, 184.0, 152.0, 37.0,
            -72.0, 212.0, 12.0, 85.0, -154.0, -241.0, 49.0, -89.0, 101.0, -22.0,
            -141.0, 196.0, -86.0, 152.0, -136.0, -216.0, 50.0, 73.0, -61.0,
            -24.0, 4.0, -113.0, -186.0, 80.0, 27.0, 200.0, -216.0, -59.0, 103.0,
            -174.0, 72.0, 65.0, 17.0, 143.0, 114.0, 49.0, 134.0, -95.0, 103.0,
            -128.0, -66.0, 179.0, 60.0, 36.0, -196.0, -133.0, 25.0, -184.0,
            71.0, 39.0, -33.0, 106.0, -171.0, 176.0, 319.0, -192.0, 192.0, -7.0,
            242.0, 47.0, 59.0, 141.0, -92.0, 151.0, 96.0, -97.0, -97.0, -223.0,
            85.0, 279.0, 30.0, 212.0, 193.0, 25.0, -79.0, 238.0, 162.0, -98.0,
            -24.0, -158.0, 98.0, 111.0, -169.0, 112.0, -50.0, 169.0, -312.0,
            -110.0, -91.0, -256.0, 20.0, 25.0, 70.0, 103.0, -321.0, 133.0,
          ],
          [
            240.0, 240.0, 212.0, 214.0, 216.0, 218.0, -20.0, -30.0, 390.0,
            282.0, 168.0, 186.0, 204.0, 222.0, -150.0, -60.0, 720.0, 319.0,
            -31.0, -21.0, -11.0, -1.0, -717.0, -288.0, 726.0, 307.0, 29.0, 39.0,
            49.0, 59.0, -663.0, -216.0, 732.0, 295.0, 89.0, 99.0, 109.0, 119.0,
            -609.0, -144.0, 498.0, -17.0, -123.0, -115.0, -107.0, -99.0, -595.0,
            -42.0, 354.0, -191.0, -499.0, -507.0, -515.0, -523.0, -891.0,
            -300.0, 390.0, 935.0, 1134.0, 1146.0, 1158.0, 1170.0, 762.0, 145.0,
            1050.0, 1036.0, 1638.0, 1655.0, 1672.0, 1689.0, 524.0, 603.0,
            1002.0, 1484.0, 1641.0, 1666.0, 1691.0, 1716.0, 583.0, 96.0, 1158.0,
            1670.0, 1791.0, 1816.0, 1841.0, 1866.0, 577.0, 60.0, 1314.0, 1856.0,
            1941.0, 1966.0, 1991.0, 2016.0, 571.0, 24.0, 930.0, 537.0, 597.0,
            610.0, 623.0, 636.0, -407.0, 53.0, -84.0, 772.0, 93.0, 101.0, 109.0,
            117.0, 185.0, -741.0, 240.0, 420.0, 392.0, 394.0, 396.0, 398.0,
            160.0, -30.0, 390.0, 822.0, 1788.0, 1806.0, 1824.0, 1842.0, 1470.0,
            1020.0, 810.0, 139.0, 869.0, 879.0, 889.0, 899.0, 93.0, 792.0,
            816.0, 127.0, 929.0, 939.0, 949.0, 959.0, 147.0, 864.0, 822.0,
            115.0, 989.0, 999.0, 1009.0, 1019.0, 201.0, 936.0, 588.0, -377.0,
            597.0, 605.0, 613.0, 621.0, 35.0, 1038.0, 444.0, -911.0, -1219.0,
            -1227.0, -1235.0, -1243.0, -1701.0, -300.0, 840.0, 2645.0, 2214.0,
            2226.0, 2238.0, 2250.0, 1392.0, -485.0, 3030.0, 2296.0, 3168.0,
            3185.0, 3202.0, 3219.0, 74.0, 873.0, 3342.0, 4274.0, 3891.0, 3916.0,
            3941.0, 3966.0, 493.0, -444.0, 3498.0, 4460.0, 4041.0, 4066.0,
            4091.0, 4116.0, 487.0, -480.0, 3654.0, 4646.0, 4191.0, 4216.0,
            4241.0, 4266.0, 481.0, -516.0, 2820.0, 1617.0, 1767.0, 1780.0,
            1793.0, 1806.0, -1127.0, 143.0, 276.0, 2302.0, 813.0, 821.0, 829.0,
            837.0, 545.0, -1551.0,
          ],
          [
            -538.0, -3412.0, -3599.0, 1191.0, -742.0, -909.0, -372.0, -1714.0,
            -1220.0, -332.0, -1990.0, -1258.0, 461.0, -3223.0, -3971.0, 2216.0,
            -740.0, -3122.0, -1108.0, -4672.0, -4469.0, 1551.0, -352.0, -219.0,
            168.0, -1234.0, -740.0, -242.0, -2650.0, -1798.0, 521.0, -4663.0,
            -6011.0, 2846.0, -980.0, -4202.0, -1678.0, -5932.0, -5339.0, 1911.0,
            38.0, 471.0, 708.0, -754.0, -260.0, -152.0, -3310.0, -2338.0, 581.0,
            -6103.0, -8051.0, 3476.0, -1220.0, -5282.0,
          ],
          [3570.0, 5370.0, 7170.0]
        );
      });
      it('forward / backward stride, padding', async () => {
        await doConv2d(
          { stride: 2, padding: 1 },
          {
            data: [
              6.0, -5.0, -9.0, -9.0, 2.0, -2.0, -4.0, 0.0, 0.0, -5.0, 4.0, -8.0,
              -6.0, -2.0, 7.0, 6.0, -8.0, 2.0, 9.0, -3.0, 1.0, 9.0, 0.0, 1.0,
              5.0, -8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0, -1.0, 9.0,
              -4.0, -9.0, -1.0, -6.0, -4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0,
              -6.0, 8.0, -1.0, 3.0, 4.0, -1.0, -9.0, 2.0, 3.0, 7.0, -4.0, 4.0,
              6.0, -7.0, 3.0, 9.0, 4.0, -4.0, 0.0, -6.0, 9.0, -8.0, -5.0, -5.0,
              -4.0, 2.0, -9.0, -3.0, 1.0, -7.0, -9.0, 5.0, 1.0, 3.0, 5.0, 1.0,
              6.0, 7.0, 1.0, 6.0, -8.0, 7.0, 7.0, 1.0, 2.0, -7.0, -7.0, -8.0,
              2.0, 1.0, -1.0, 4.0, -4.0, -1.0, -2.0, -7.0, -1.0, -4.0, 0.0, 7.0,
              -1.0, -4.0, 7.0, 7.0, 5.0, -4.0, -8.0, 1.0, 4.0, -6.0, 4.0, -8.0,
              5.0, 3.0, 4.0, 4.0, 5.0, -6.0, 1.0, 4.0, 4.0, 2.0, -2.0, 3.0,
              -5.0, -9.0, 5.0, -2.0, 9.0, 7.0, -8.0, 6.0, -9.0, -9.0, -9.0,
              -6.0, 5.0, -8.0, 9.0, -9.0, 1.0, 1.0, -2.0, 8.0, 3.0, -6.0, 7.0,
              -1.0, 3.0, 3.0, 8.0, -8.0, 6.0, -6.0, 3.0, 1.0, -9.0, 2.0, 9.0,
              2.0, -5.0, -7.0, -8.0, 9.0, 4.0, 7.0, 8.0, 5.0, 1.0, -4.0, 0.0,
              -5.0, -4.0, 9.0, -7.0, 6.0, 4.0, -6.0, -6.0, 7.0, -3.0, 3.0, 0.0,
              -8.0, -1.0, 5.0, -7.0, -4.0, -9.0, 4.0, -9.0, 8.0, 5.0, 7.0, -7.0,
              9.0, 2.0, -6.0, 0.0, -4.0, 4.0, 8.0, -4.0, -2.0, -1.0, 2.0, -4.0,
              6.0, 4.0, 4.0, 7.0, -3.0, 7.0, -5.0, 0.0, -6.0,
            ],
            shape: [2, 2, 7, 8],
          },
          {
            data: [
              -1.0, 1.0, 4.0, -5.0, 9.0, 9.0, -5.0, -4.0, 3.0, -2.0, 5.0, -7.0,
              4.0, -8.0, 4.0, 6.0, 1.0, 1.0, -6.0, 2.0, -7.0, 5.0, -6.0, 9.0,
              1.0, 1.0, 4.0, 1.0, 0.0, -6.0, 5.0, -6.0, -3.0, 2.0, 6.0, -4.0,
              7.0, -1.0, 3.0, 0.0, 1.0, -6.0, 5.0, -6.0, -7.0, 6.0, 9.0, 6.0,
              8.0, -8.0, 9.0, -4.0, 6.0, -6.0,
            ],
            shape: [3, 2, 3, 3],
          },
          { data: [7.0, 7.0, 4.0], shape: [3] },
          [2, 3, 4, 4],
          [
            5.0, -99.0, 15.0, -93.0, -19.0, 224.0, 140.0, 126.0, 135.0, 58.0,
            -198.0, -43.0, 24.0, -17.0, -10.0, -38.0, -93.0, -15.0, -11.0,
            -56.0, 117.0, 184.0, 180.0, 95.0, 133.0, 119.0, -90.0, -81.0, 105.0,
            -10.0, 40.0, 50.0, 162.0, 232.0, 75.0, -61.0, 45.0, -107.0, -213.0,
            -208.0, 116.0, -106.0, 184.0, 37.0, -44.0, 56.0, 48.0, -134.0, 14.0,
            -12.0, -67.0, 76.0, 44.0, 152.0, -216.0, 73.0, -53.0, 200.0, -59.0,
            -174.0, -66.0, 8.0, -148.0, -41.0, 31.0, 65.0, -151.0, -184.0, 19.0,
            36.0, -133.0, -184.0, -236.0, -192.0, -7.0, 47.0, -119.0, 32.0,
            19.0, -6.0, -11.0, 88.0, 121.0, -85.0, -240.0, 25.0, 238.0, -98.0,
            113.0, 169.0, -110.0, -256.0, 113.0, -102.0, -152.0, -92.0,
          ],
          [
            -64.0, 32.0, -60.0, 44.0, -56.0, 56.0, -52.0, -12.0, -168.0, 129.0,
            -175.0, 130.0, -182.0, 131.0, -189.0, -176.0, -48.0, 80.0, -44.0,
            92.0, -40.0, 104.0, -36.0, 36.0, -196.0, 133.0, -203.0, 134.0,
            -210.0, 135.0, -217.0, -176.0, -32.0, 128.0, -28.0, 140.0, -24.0,
            152.0, -20.0, 84.0, -224.0, 137.0, -231.0, 138.0, -238.0, 139.0,
            -245.0, -176.0, -16.0, 176.0, -12.0, 188.0, -8.0, 200.0, -4.0,
            132.0, -352.0, 593.0, -374.0, 620.0, -396.0, 647.0, -418.0, 270.0,
            632.0, -47.0, 659.0, -54.0, 686.0, -61.0, 713.0, -236.0, -440.0,
            701.0, -462.0, 728.0, -484.0, 755.0, -506.0, 310.0, 740.0, -75.0,
            767.0, -82.0, 794.0, -89.0, 821.0, -300.0, -528.0, 809.0, -550.0,
            836.0, -572.0, 863.0, -594.0, 350.0, 848.0, -103.0, 875.0, -110.0,
            902.0, -117.0, 929.0, -364.0, -616.0, 917.0, -638.0, 944.0, -660.0,
            971.0, -682.0, 390.0, 128.0, 608.0, 132.0, 620.0, 136.0, 632.0,
            140.0, 564.0, -504.0, 177.0, -511.0, 178.0, -518.0, 179.0, -525.0,
            -176.0, 144.0, 656.0, 148.0, 668.0, 152.0, 680.0, 156.0, 612.0,
            -532.0, 181.0, -539.0, 182.0, -546.0, 183.0, -553.0, -176.0, 160.0,
            704.0, 164.0, 716.0, 168.0, 728.0, 172.0, 660.0, -560.0, 185.0,
            -567.0, 186.0, -574.0, 187.0, -581.0, -176.0, 176.0, 752.0, 180.0,
            764.0, 184.0, 776.0, 188.0, 708.0, -1408.0, 1889.0, -1430.0, 1916.0,
            -1452.0, 1943.0, -1474.0, 750.0, 1928.0, -383.0, 1955.0, -390.0,
            1982.0, -397.0, 2009.0, -1004.0, -1496.0, 1997.0, -1518.0, 2024.0,
            -1540.0, 2051.0, -1562.0, 790.0, 2036.0, -411.0, 2063.0, -418.0,
            2090.0, -425.0, 2117.0, -1068.0, -1584.0, 2105.0, -1606.0, 2132.0,
            -1628.0, 2159.0, -1650.0, 830.0, 2144.0, -439.0, 2171.0, -446.0,
            2198.0, -453.0, 2225.0, -1132.0, -1672.0, 2213.0, -1694.0, 2240.0,
            -1716.0, 2267.0, -1738.0, 870.0,
          ],
          [
            -55.0, 1300.0, -552.0, -1583.0, 1430.0, -1507.0, 1.0, 1132.0,
            -508.0, 848.0, -2291.0, 1944.0, 918.0, 722.0, 183.0, 808.0, -2191.0,
            1848.0, -279.0, 1972.0, -728.0, -1983.0, 1574.0, -1955.0, -223.0,
            1804.0, -684.0, 1008.0, -2691.0, 2328.0, 1158.0, 690.0, 423.0,
            968.0, -2591.0, 2232.0, -503.0, 2644.0, -904.0, -2383.0, 1718.0,
            -2403.0, -447.0, 2476.0, -860.0, 1168.0, -3091.0, 2712.0, 1398.0,
            658.0, 663.0, 1128.0, -2991.0, 2616.0,
          ],
          [1008.0, 1520.0, 2032.0]
        );
      });
      it('forward / backward padding, dilation', async () => {
        await doConv2d(
          { padding: 1, dilation: 2 },
          {
            data: [
              6.0, -5.0, -9.0, -9.0, 2.0, -2.0, -4.0, 0.0, 0.0, -5.0, 4.0, -8.0,
              -6.0, -2.0, 7.0, 6.0, -8.0, 2.0, 9.0, -3.0, 1.0, 9.0, 0.0, 1.0,
              5.0, -8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0, -1.0, 9.0,
              -4.0, -9.0, -1.0, -6.0, -4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0,
              -6.0, 8.0, -1.0, 3.0, 4.0, -1.0, -9.0, 2.0, 3.0, 7.0, -4.0, 4.0,
              6.0, -7.0, 3.0, 9.0, 4.0, -4.0, 0.0, -6.0, 9.0, -8.0, -5.0, -5.0,
              -4.0, 2.0, -9.0, -3.0, 1.0, -7.0, -9.0, 5.0, 1.0, 3.0, 5.0, 1.0,
              6.0, 7.0, 1.0, 6.0, -8.0, 7.0, 7.0, 1.0, 2.0, -7.0, -7.0, -8.0,
              2.0, 1.0, -1.0, 4.0, -4.0, -1.0, -2.0, -7.0, -1.0, -4.0, 0.0, 7.0,
              -1.0, -4.0, 7.0, 7.0, 5.0, -4.0, -8.0, 1.0, 4.0, -6.0, 4.0, -8.0,
              5.0, 3.0, 4.0, 4.0, 5.0, -6.0, 1.0, 4.0, 4.0, 2.0, -2.0, 3.0,
              -5.0, -9.0, 5.0, -2.0, 9.0, 7.0, -8.0, 6.0, -9.0, -9.0, -9.0,
              -6.0, 5.0, -8.0, 9.0, -9.0, 1.0, 1.0, -2.0, 8.0, 3.0, -6.0, 7.0,
              -1.0, 3.0, 3.0, 8.0, -8.0, 6.0, -6.0, 3.0, 1.0, -9.0, 2.0, 9.0,
              2.0, -5.0, -7.0, -8.0, 9.0, 4.0, 7.0, 8.0, 5.0, 1.0, -4.0, 0.0,
              -5.0, -4.0, 9.0, -7.0, 6.0, 4.0, -6.0, -6.0, 7.0, -3.0, 3.0, 0.0,
              -8.0, -1.0, 5.0, -7.0, -4.0, -9.0, 4.0, -9.0, 8.0, 5.0, 7.0, -7.0,
              9.0, 2.0, -6.0, 0.0, -4.0, 4.0, 8.0, -4.0, -2.0, -1.0, 2.0, -4.0,
              6.0, 4.0, 4.0, 7.0, -3.0, 7.0, -5.0, 0.0, -6.0,
            ],
            shape: [2, 2, 7, 8],
          },
          {
            data: [
              -1.0, 1.0, 4.0, -5.0, 9.0, 9.0, -5.0, -4.0, 3.0, -2.0, 5.0, -7.0,
              4.0, -8.0, 4.0, 6.0, 1.0, 1.0, -6.0, 2.0, -7.0, 5.0, -6.0, 9.0,
              1.0, 1.0, 4.0, 1.0, 0.0, -6.0, 5.0, -6.0, -3.0, 2.0, 6.0, -4.0,
              7.0, -1.0, 3.0, 0.0, 1.0, -6.0, 5.0, -6.0, -7.0, 6.0, 9.0, 6.0,
              8.0, -8.0, 9.0, -4.0, 6.0, -6.0,
            ],
            shape: [3, 2, 3, 3],
          },
          { data: [7.0, 7.0, 4.0], shape: [3] },
          [2, 3, 5, 6],
          [
            -48.0, 44.0, -16.0, -60.0, 107.0, 156.0, -49.0, 70.0, 46.0, -42.0,
            41.0, 104.0, -37.0, -92.0, 127.0, 190.0, 49.0, 38.0, 82.0, -107.0,
            -41.0, -3.0, -153.0, 51.0, 93.0, 15.0, -71.0, 5.0, -30.0, 79.0,
            37.0, -44.0, 34.0, 167.0, 112.0, 41.0, 73.0, -146.0, 266.0, 131.0,
            -73.0, 14.0, 178.0, 13.0, 11.0, -18.0, 91.0, 43.0, 24.0, -46.0,
            -33.0, -13.0, 91.0, 75.0, -51.0, -197.0, -38.0, 98.0, 10.0, -73.0,
            43.0, 131.0, -172.0, -232.0, -66.0, 82.0, -11.0, 232.0, -18.0,
            -133.0, 181.0, 101.0, -204.0, 57.0, -30.0, 115.0, -144.0, -48.0,
            -238.0, -278.0, 150.0, -68.0, -142.0, 127.0, 69.0, 183.0, 6.0, 94.0,
            127.0, 64.0, 73.0, -11.0, -22.0, 34.0, -123.0, 49.0, 51.0, -112.0,
            -10.0, -70.0, -83.0, 112.0, 129.0, -145.0, 1.0, -125.0, -190.0,
            -81.0, -203.0, 82.0, -54.0, 20.0, 202.0, 59.0, -44.0, 103.0, 16.0,
            42.0, -100.0, 65.0, 141.0, 27.0, -126.0, -34.0, -41.0, -42.0, -70.0,
            -66.0, -65.0, 32.0, 138.0, -72.0, 55.0, 61.0, -63.0, 51.0, -182.0,
            -22.0, -120.0, 72.0, 75.0, 28.0, 153.0, 85.0, -12.0, 100.0, 70.0,
            234.0, -192.0, 20.0, -60.0, 122.0, 127.0, 77.0, 34.0, 20.0, 196.0,
            -48.0, 87.0, 26.0, -52.0, 114.0, -55.0, -37.0, 55.0, -73.0, 189.0,
            -111.0, 141.0, 147.0, -106.0, -92.0, 83.0, -167.0, -46.0, -31.0,
            117.0, -352.0, 170.0, -153.0,
          ],
          [
            240.0, 252.0, 254.0, 226.0, 228.0, -10.0, -8.0, -30.0, 390.0, 294.0,
            300.0, 186.0, 204.0, -168.0, -150.0, -72.0, 390.0, 330.0, 336.0,
            294.0, 312.0, -60.0, -42.0, 0.0, 721.0, 368.0, 366.0, 88.0, 98.0,
            -618.0, -609.0, -228.0, 487.0, 56.0, 52.0, -126.0, -118.0, -602.0,
            -595.0, -126.0, 493.0, 32.0, 28.0, -78.0, -70.0, -560.0, -553.0,
            -54.0, 349.0, -142.0, -150.0, -458.0, -466.0, -828.0, -837.0,
            -300.0, 425.0, 1054.0, 1073.0, 1230.0, 1242.0, 804.0, 811.0, 110.0,
            1102.0, 1172.0, 1186.0, 1746.0, 1763.0, 568.0, 563.0, 558.0, 1234.0,
            1256.0, 1270.0, 1848.0, 1865.0, 538.0, 533.0, 576.0, 1190.0, 1708.0,
            1739.0, 1872.0, 1897.0, 602.0, 601.0, 78.0, 801.0, 384.0, 396.0,
            504.0, 517.0, -376.0, -384.0, 100.0, 927.0, 456.0, 468.0, 582.0,
            595.0, -424.0, -432.0, 106.0, -104.0, 674.0, 691.0, 66.0, 74.0,
            166.0, 170.0, -678.0, 240.0, 432.0, 434.0, 406.0, 408.0, 170.0,
            172.0, -30.0, 390.0, 834.0, 840.0, 1806.0, 1824.0, 1452.0, 1470.0,
            1008.0, 390.0, 870.0, 876.0, 1914.0, 1932.0, 1560.0, 1578.0, 1080.0,
            811.0, 188.0, 186.0, 988.0, 998.0, 192.0, 201.0, 852.0, 577.0,
            -304.0, -308.0, 594.0, 602.0, 28.0, 35.0, 954.0, 583.0, -328.0,
            -332.0, 642.0, 650.0, 70.0, 77.0, 1026.0, 439.0, -862.0, -870.0,
            -1178.0, -1186.0, -1638.0, -1647.0, -300.0, 875.0, 2764.0, 2783.0,
            2310.0, 2322.0, 1434.0, 1441.0, -520.0, 3082.0, 2432.0, 2446.0,
            3276.0, 3293.0, 118.0, 113.0, 828.0, 3214.0, 2516.0, 2530.0, 3378.0,
            3395.0, 88.0, 83.0, 846.0, 3530.0, 4498.0, 4529.0, 4122.0, 4147.0,
            512.0, 511.0, -462.0, 2691.0, 1464.0, 1476.0, 1674.0, 1687.0,
            -1096.0, -1104.0, 190.0, 2817.0, 1536.0, 1548.0, 1752.0, 1765.0,
            -1144.0, -1152.0, 196.0, 256.0, 2204.0, 2221.0, 786.0, 794.0, 526.0,
            530.0, -1488.0,
          ],
          [
            2668.0, -2626.0, -3017.0, 3435.0, -742.0, -2240.0, 595.0, -1218.0,
            -1615.0, -3900.0, -2757.0, 2074.0, -491.0, -3223.0, -1531.0, 2884.0,
            -348.0, -1456.0, 2848.0, -3616.0, -3497.0, 4575.0, -352.0, -2450.0,
            1555.0, -1098.0, -1855.0, -4410.0, -3657.0, 2944.0, -311.0, -4663.0,
            -2911.0, 4444.0, 312.0, -1666.0, 3028.0, -4606.0, -3977.0, 5715.0,
            38.0, -2660.0, 2515.0, -978.0, -2095.0, -4920.0, -4557.0, 3814.0,
            -131.0, -6103.0, -4291.0, 6004.0, 972.0, -1876.0,
          ],
          [3570.0, 5370.0, 7170.0]
        );
      });
      it('forward / backward groups', async () => {
        await doConv2d(
          { groups: 2 },
          {
            data: [
              6.0, -5.0, -9.0, -9.0, 2.0, -2.0, -4.0, 0.0, 0.0, -5.0, 4.0, -8.0,
              -6.0, -2.0, 7.0, 6.0, -8.0, 2.0, 9.0, -3.0, 1.0, 9.0, 0.0, 1.0,
              5.0, -8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0, -1.0, 9.0,
              -4.0, -9.0, -1.0, -6.0, -4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0,
              -6.0, 8.0, -1.0, 3.0, 4.0, -1.0, -9.0, 2.0, 3.0, 7.0, -4.0, 4.0,
              6.0, -7.0, 3.0, 9.0, 4.0, -4.0, 0.0, -6.0, 9.0, -8.0, -5.0, -5.0,
              -4.0, 2.0, -9.0, -3.0, 1.0, -7.0, -9.0, 5.0, 1.0, 3.0, 5.0, 1.0,
              6.0, 7.0, 1.0, 6.0, -8.0, 7.0, 7.0, 1.0, 2.0, -7.0, -7.0, -8.0,
              2.0, 1.0, -1.0, 4.0, -4.0, -1.0, -2.0, -7.0, -1.0, -4.0, 0.0, 7.0,
              -1.0, -4.0, 7.0, 7.0, 5.0, -4.0, -8.0, 1.0, 4.0, -6.0, 4.0, -8.0,
              5.0, 3.0, 4.0, 4.0, 5.0, -6.0, 1.0, 4.0, 4.0, 2.0, -2.0, 3.0,
              -5.0, -9.0, 5.0, -2.0, 9.0, 7.0, -8.0, 6.0, -9.0, -9.0, -9.0,
              -6.0, 5.0, -8.0, 9.0, -9.0, 1.0, 1.0, -2.0, 8.0, 3.0, -6.0, 7.0,
              -1.0, 3.0, 3.0, 8.0, -8.0, 6.0, -6.0, 3.0, 1.0, -9.0, 2.0, 9.0,
              2.0, -5.0, -7.0, -8.0, 9.0, 4.0, 7.0, 8.0, 5.0, 1.0, -4.0, 0.0,
              -5.0, -4.0, 9.0, -7.0, 6.0, 4.0, -6.0, -6.0, 7.0, -3.0, 3.0, 0.0,
              -8.0, -1.0, 5.0, -7.0, -4.0, -9.0, 4.0, -9.0, 8.0, 5.0, 7.0, -7.0,
              9.0, 2.0, -6.0, 0.0, -4.0, 4.0, 8.0, -4.0, -2.0, -1.0, 2.0, -4.0,
              6.0, 4.0, 4.0, 7.0, -3.0, 7.0, -5.0, 0.0, -6.0, -1.0, 1.0, 4.0,
              -5.0, 9.0, 9.0, -5.0, -4.0, 3.0, -2.0, 5.0, -7.0, 4.0, -8.0, 4.0,
              6.0, 1.0, 1.0, -6.0, 2.0, -7.0, 5.0, -6.0, 9.0, 1.0, 1.0, 4.0,
              1.0, 0.0, -6.0, 5.0, -6.0, -3.0, 2.0, 6.0, -4.0, 7.0, -1.0, 3.0,
              0.0, 1.0, -6.0, 5.0, -6.0, -7.0, 6.0, 9.0, 6.0, 8.0, -8.0, 9.0,
              -4.0, 6.0, -6.0, 7.0, 7.0, 4.0, 7.0, 4.0, 9.0, 6.0, -7.0, -8.0,
              8.0, 5.0, 8.0, 1.0, -3.0, 9.0, -4.0, 4.0, 2.0, 4.0, -1.0, -4.0,
              -8.0, -3.0, -7.0, -4.0, -8.0, -8.0, -4.0, 2.0, -3.0, 5.0, 7.0,
              5.0, -5.0, 8.0, -8.0, -3.0, 7.0, -7.0, 4.0, -7.0, -1.0, 8.0, 0.0,
              5.0, -4.0, 7.0, 9.0, 0.0, -8.0, 7.0, 3.0, 5.0, -2.0, 6.0, 3.0,
              9.0, -3.0, -9.0, 4.0, 7.0, 2.0, -1.0, -9.0, -5.0, -6.0, 3.0, 0.0,
              -1.0, 9.0, -6.0, -8.0, -2.0, -5.0, -3.0, -6.0, -8.0, -1.0, -5.0,
              -5.0, 5.0, 3.0, -9.0, 8.0, 2.0, 3.0, -6.0, 7.0, -2.0, 3.0, 1.0,
              -7.0, 0.0, -1.0, -1.0, 7.0, 0.0, 1.0, 4.0, 1.0, -5.0, -5.0, 6.0,
              -3.0, 3.0, -6.0, -2.0, 5.0, -7.0, -2.0, 6.0, 3.0, 3.0, 7.0, 9.0,
              3.0, 0.0, 2.0, -9.0, 8.0, -1.0, 1.0, 6.0, 0.0, 7.0, -4.0, 4.0,
              8.0, 6.0, -2.0, -7.0, -1.0, -4.0, -1.0, -1.0, -4.0, 2.0, 4.0, 2.0,
              3.0, -1.0, -4.0, -4.0, 3.0, -9.0, 5.0, 1.0, -7.0, -4.0, -8.0,
              -1.0, 3.0, 3.0, -7.0, 8.0, -1.0, 4.0, 3.0, 9.0, 1.0, -2.0, -7.0,
              1.0, -9.0, 1.0, 5.0, 3.0, -3.0, 0.0, 9.0,
            ],
            shape: [2, 4, 7, 8],
          },
          {
            data: [
              9.0, 4.0, 4.0, -4.0, 8.0, -8.0, 5.0, -5.0, -3.0, -3.0, 0.0, 7.0,
              -7.0, -9.0, -7.0, 2.0, 1.0, -6.0, 5.0, 3.0, -6.0, -5.0, -9.0, 7.0,
              -4.0, 7.0, -3.0, -7.0, -4.0, -8.0, -1.0, -8.0, -8.0, 0.0, -2.0,
              4.0,
            ],
            shape: [2, 2, 3, 3],
          },
          { data: [-8.0, 5.0], shape: [2] },
          [2, 2, 5, 6],
          [
            -140.0, 55.0, -74.0, -143.0, -6.0, 183.0, 62.0, 19.0, -15.0, -170.0,
            97.0, -10.0, -316.0, -56.0, -18.0, 9.0, -158.0, 62.0, -11.0, -333.0,
            173.0, 222.0, 79.0, 301.0, -23.0, 31.0, 44.0, -29.0, -116.0, -55.0,
            192.0, -6.0, 1.0, -221.0, -177.0, 7.0, 63.0, -1.0, 54.0, -11.0,
            130.0, -41.0, -205.0, 335.0, -6.0, 42.0, 65.0, 60.0, 160.0, -46.0,
            92.0, 186.0, 89.0, 120.0, -58.0, -31.0, 93.0, -180.0, 99.0, -106.0,
            -114.0, 170.0, -94.0, 88.0, -183.0, 166.0, 40.0, -54.0, 288.0,
            -98.0, 243.0, -1.0, -26.0, -86.0, -49.0, -121.0, -252.0, 54.0, 82.0,
            110.0, -4.0, 33.0, -59.0, -19.0, -191.0, 136.0, -23.0, -239.0,
            -91.0, -84.0, -249.0, 20.0, 29.0, -145.0, 0.0, -38.0, 64.0, 47.0,
            55.0, 29.0, 14.0, -131.0, 19.0, -5.0, 26.0, 148.0, 6.0, -67.0,
            123.0, 159.0, 190.0, 92.0, -73.0, 19.0, 32.0, 10.0, 86.0, -119.0,
            3.0, 98.0,
          ],
          [
            0.0, 9.0, 22.0, 39.0, 56.0, 73.0, 36.0, 20.0, 54.0, 83.0, 124.0,
            137.0, 150.0, 163.0, 92.0, 4.0, 84.0, 190.0, 207.0, 217.0, 227.0,
            237.0, 103.0, -35.0, 144.0, 292.0, 267.0, 277.0, 287.0, 297.0,
            103.0, -77.0, 204.0, 394.0, 327.0, 337.0, 347.0, 357.0, 103.0,
            -119.0, -6.0, 97.0, -145.0, -152.0, -159.0, -166.0, -173.0, -301.0,
            120.0, 5.0, -67.0, -70.0, -73.0, -76.0, -229.0, -87.0, 0.0, -3.0,
            -6.0, -2.0, 2.0, 6.0, 28.0, 35.0, -18.0, -28.0, -5.0, -24.0, -43.0,
            -62.0, -3.0, 42.0, -78.0, -140.0, -114.0, -136.0, -158.0, -180.0,
            -76.0, 12.0, -126.0, -236.0, -246.0, -268.0, -290.0, -312.0, -160.0,
            -24.0, -174.0, -332.0, -378.0, -400.0, -422.0, -444.0, -244.0,
            -60.0, -132.0, -335.0, -624.0, -650.0, -676.0, -702.0, -566.0,
            -341.0, 48.0, 74.0, -67.0, -70.0, -73.0, -76.0, -139.0, -174.0,
            150.0, 245.0, 73.0, 75.0, 77.0, 79.0, -99.0, -210.0, 30.0, -132.0,
            -144.0, -149.0, -154.0, -159.0, -194.0, -1.0, -90.0, -82.0, -175.0,
            -180.0, -185.0, -190.0, -81.0, -100.0, -114.0, -100.0, -205.0,
            -210.0, -215.0, -220.0, -87.0, -112.0, -138.0, -118.0, -235.0,
            -240.0, -245.0, -250.0, -93.0, -124.0, -462.0, -621.0, -398.0,
            -405.0, -412.0, -419.0, 90.0, 254.0, -216.0, 158.0, -1.0, -1.0,
            -1.0, -1.0, 239.0, -177.0, -210.0, -337.0, -588.0, -607.0, -626.0,
            -645.0, -412.0, -280.0, -282.0, -674.0, -1222.0, -1258.0, -1294.0,
            -1330.0, -1036.0, -608.0, -330.0, -854.0, -1380.0, -1414.0, -1448.0,
            -1482.0, -1138.0, -564.0, -378.0, -986.0, -1584.0, -1618.0, -1652.0,
            -1686.0, -1294.0, -636.0, -426.0, -1118.0, -1788.0, -1822.0,
            -1856.0, -1890.0, -1450.0, -708.0, -54.0, -583.0, -834.0, -849.0,
            -864.0, -879.0, -834.0, -260.0, 0.0, -108.0, 106.0, 108.0, 110.0,
            112.0, 114.0, 236.0, 540.0, 789.0, 1042.0, 1059.0, 1076.0, 1093.0,
            516.0, 260.0, 354.0, 1103.0, 904.0, 917.0, 930.0, 943.0, 572.0,
            -236.0, 684.0, 1210.0, 807.0, 817.0, 827.0, 837.0, 103.0, -455.0,
            744.0, 1312.0, 867.0, 877.0, 887.0, 897.0, 103.0, -497.0, 804.0,
            1414.0, 927.0, 937.0, 947.0, 957.0, 103.0, -539.0, 54.0, 337.0,
            -565.0, -572.0, -579.0, -586.0, -653.0, -961.0, 420.0, 5.0, -247.0,
            -250.0, -253.0, -256.0, -709.0, -267.0, -180.0, -183.0, 234.0,
            238.0, 242.0, 246.0, 448.0, 455.0, -618.0, -1168.0, -1145.0,
            -1164.0, -1183.0, -1202.0, -543.0, 42.0, -558.0, -1100.0, -1434.0,
            -1456.0, -1478.0, -1500.0, -916.0, -348.0, -606.0, -1196.0, -1566.0,
            -1588.0, -1610.0, -1632.0, -1000.0, -384.0, -654.0, -1292.0,
            -1698.0, -1720.0, -1742.0, -1764.0, -1084.0, -420.0, -432.0,
            -1115.0, -2184.0, -2210.0, -2236.0, -2262.0, -1826.0, -1121.0,
            168.0, 254.0, -247.0, -250.0, -253.0, -256.0, -439.0, -534.0, 450.0,
            725.0, 193.0, 195.0, 197.0, 199.0, -279.0, -570.0, 30.0, -492.0,
            -444.0, -449.0, -454.0, -459.0, -494.0, 59.0, -330.0, -262.0,
            -475.0, -480.0, -485.0, -490.0, -141.0, -220.0, -354.0, -280.0,
            -505.0, -510.0, -515.0, -520.0, -147.0, -232.0, -378.0, -298.0,
            -535.0, -540.0, -545.0, -550.0, -153.0, -244.0, -1002.0, -1281.0,
            -818.0, -825.0, -832.0, -839.0, 210.0, 494.0, -456.0, 338.0, -1.0,
            -1.0, -1.0, -1.0, 479.0, -357.0, -630.0, -997.0, -1728.0, -1747.0,
            -1766.0, -1785.0, -1132.0, -760.0, -762.0, -1874.0, -3382.0,
            -3418.0, -3454.0, -3490.0, -2716.0, -1568.0, -810.0, -2174.0,
            -3420.0, -3454.0, -3488.0, -3522.0, -2698.0, -1284.0, -858.0,
            -2306.0, -3624.0, -3658.0, -3692.0, -3726.0, -2854.0, -1356.0,
            -906.0, -2438.0, -3828.0, -3862.0, -3896.0, -3930.0, -3010.0,
            -1428.0, -114.0, -1243.0, -1734.0, -1749.0, -1764.0, -1779.0,
            -1674.0, -500.0, 0.0, -228.0, 226.0, 228.0, 230.0, 232.0, 234.0,
            476.0,
          ],
          [
            1263.0, 1603.0, 1756.0, -488.0, 948.0, 2335.0, 426.0, 1378.0,
            3299.0, 1269.0, -584.0, -1151.0, 2061.0, 328.0, -1228.0, 2853.0,
            1770.0, 492.0, -3735.0, -3747.0, -3836.0, -2480.0, -2502.0, -2546.0,
            -2279.0, -890.0, 175.0, -741.0, -2416.0, -1814.0, 1312.0, -1350.0,
            -1771.0, 31.0, -2502.0, -1468.0,
          ],
          [2670.0, 4470.0]
        );
      });
    });
  });
}

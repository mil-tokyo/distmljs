import { assert } from 'chai';
import { DType } from '../../dtype';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import { WebGPUTensor } from '../../tensor/webgpu/webgpuTensor';
import { testFlag } from '../testFlag';
import { arrayNearlyEqual } from '../testUtil';

/**
 * 同じ入力に対するCPU実装の結果を期待値として、WebGPU実装を検証する。
 */
async function assertEqualsCPU(
  inputs: { data: number[]; shape: number[]; dtype?: DType }[],
  webgpuOp: (...xs: WebGPUTensor[]) => WebGPUTensor,
  cpuOp: (...xs: CPUTensor[]) => CPUTensor,
  message?: string
): Promise<void> {
  const cpuInputs = inputs.map((i) =>
    CPUTensor.fromArray(i.data, i.shape, i.dtype)
  );
  const webgpuInputs = inputs.map((i) =>
    WebGPUTensor.fromArray(i.data, i.shape, i.dtype)
  );
  const expected = cpuOp(...cpuInputs);
  const actual = webgpuOp(...webgpuInputs);
  assert.deepEqual(actual.shape, expected.shape, `${message}: shape`);
  assert.equal(actual.dtype, expected.dtype, `${message}: dtype`);
  arrayNearlyEqual(
    await actual.toArrayAsync(),
    await expected.toArrayAsync(),
    message
  );
}

describe('webgpuTensor', () => {
  if (!testFlag.webgpu) {
    return;
  }

  describe('basic', () => {
    it('computes size', async () => {
      const t = WebGPUTensor.zeros([3, 4]);
      assert.equal(t.size, 12);
      assert.equal(t.ndim, 2);
      assert.deepEqual(t.shape, [3, 4]);
      assert.deepEqual(t.strides, [4, 1]);
    });

    it('computes size of scalar', async () => {
      const t = WebGPUTensor.zeros([]);
      assert.equal(t.size, 1);
      assert.equal(t.ndim, 0);
      assert.deepEqual(t.shape, []);
      assert.deepEqual(t.strides, []);
    });

    it('create from array', async () => {
      const t = WebGPUTensor.fromArray([10, 20, 30, 4.5, 50, 60], [2, 3]);
      // WebGPUの仕様上の制限により、fromArrayでCPUから書き込んだデータを、直接読むことができない。
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 4.5);
    });

    it('create from array int32', async () => {
      // float32では正確に表せない数値
      const t = WebGPUTensor.fromArray(
        [10, 20, 30, 16843009, 16843010, 16843011],
        [2, 3],
        'int32'
      );
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 16843009);
      assert.equal(cpu.get(1, 1), 16843010);
      assert.equal(cpu.get(1, 2), 16843011);
    });

    it('create from array uint8', async () => {
      const t = WebGPUTensor.fromArray(
        [10, 20, 30, 40, 50, 60],
        [2, 3],
        'uint8'
      );
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 40);
    });

    it('create from array bool', async () => {
      const t = WebGPUTensor.fromArray([0, 1, 0, 1, 1, 0], [2, 3], 'bool');
      const cpu = await t.copy().to('cpu');
      assert.equal(cpu.get(1, 0), 1);
    });
  });

  describe('exp', () => {
    it('exp', async () => {
      const x = WebGPUTensor.fromArray([1, -1], [2]);
      const y = WebGPUTensor.exp(x);
      arrayNearlyEqual(await y.toArrayAsync(), [2.71828182845904, 0.367879441]);
    });
  });

  describe('abs', () => {
    it('abs', async () => {
      const x = WebGPUTensor.fromArray([1.5, -3.5], [2]);
      const y = WebGPUTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [1.5, 3.5]);
    });

    it('abs int32', async () => {
      const x = WebGPUTensor.fromArray([16843010, -16843009], [2], 'int32');
      const y = WebGPUTensor.abs(x);
      arrayNearlyEqual(await y.toArrayAsync(), [16843010, 16843009]);
    });
  });

  describe('add', () => {
    it('add', async () => {
      const lhs = WebGPUTensor.fromArray([10, 20], [2]);
      const rhs = WebGPUTensor.fromArray([50, 60], [2]);
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [60, 80]);
    });

    it('add int32', async () => {
      const lhs = WebGPUTensor.fromArray([16843009, 16843010], [2], 'int32');
      const rhs = WebGPUTensor.fromArray([1, -3], [2], 'int32');
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [16843010, 16843007]);
    });

    it('add uint8', async () => {
      const lhs = WebGPUTensor.fromArray([100, 101], [2], 'uint8');
      const rhs = WebGPUTensor.fromArray([1, 3], [2], 'uint8');
      const y = WebGPUTensor.add(lhs, rhs);
      assert.deepEqual(await y.toArrayAsync(), [101, 104]);
    });
  });

  describe('ravel', () => {
    it('from 2d', async () => {
      const x = WebGPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGPUTensor.ravel(x);
      assert.isTrue(x.buffer.gpuBuffer === y.buffer.gpuBuffer);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.copy().toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('flatten', () => {
    it('from 2d', async () => {
      const x = WebGPUTensor.fromArray([0, 1, 2, 3, 4, 5], [2, 3]);
      const y = WebGPUTensor.flatten(x);
      assert.isFalse(x.buffer.gpuBuffer === y.buffer.gpuBuffer);
      assert.deepEqual(y.shape, [6]);
      assert.deepEqual(await y.toArrayAsync(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('minimum / maximum / equal', () => {
    const lhs = { data: [1, -2, 3, 3.5, -5, 6], shape: [2, 3] };
    const rhs = { data: [2, -1, 3, 0.5, -6, 7], shape: [2, 3] };

    it('minimum', async () => {
      await assertEqualsCPU(
        [lhs, rhs],
        (a, b) => WebGPUTensor.minimum(a, b),
        (a, b) => CPUTensor.minimum(a, b),
        'minimum'
      );
    });

    it('maximum', async () => {
      await assertEqualsCPU(
        [lhs, rhs],
        (a, b) => WebGPUTensor.maximum(a, b),
        (a, b) => CPUTensor.maximum(a, b),
        'maximum'
      );
    });

    it('equal', async () => {
      await assertEqualsCPU(
        [lhs, rhs],
        (a, b) => WebGPUTensor.equal(a, b),
        (a, b) => CPUTensor.equal(a, b),
        'equal'
      );
    });

    it('minimum int32', async () => {
      await assertEqualsCPU(
        [
          { ...lhs, data: [1, -2, 3, 4, -5, 6], dtype: 'int32' },
          { ...rhs, data: [2, -1, 3, 0, -6, 7], dtype: 'int32' },
        ],
        (a, b) => WebGPUTensor.minimum(a, b),
        (a, b) => CPUTensor.minimum(a, b),
        'minimum int32'
      );
    });

    it('clamp', async () => {
      await assertEqualsCPU(
        [lhs, rhs],
        (a, b) => WebGPUTensor.clamp(a, undefined, b),
        (a, b) => CPUTensor.clamp(a, undefined, b),
        'clamp'
      );
    });
  });

  describe('tril / triu', () => {
    const x = {
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      shape: [3, 4],
    };

    for (const diagonal of [-1, 0, 2]) {
      it(`tril diagonal=${diagonal}`, async () => {
        await assertEqualsCPU(
          [x],
          (a) => WebGPUTensor.tril(a, diagonal),
          (a) => CPUTensor.tril(a, diagonal),
          `tril ${diagonal}`
        );
      });

      it(`triu diagonal=${diagonal}`, async () => {
        await assertEqualsCPU(
          [x],
          (a) => WebGPUTensor.triu(a, diagonal),
          (a) => CPUTensor.triu(a, diagonal),
          `triu ${diagonal}`
        );
      });
    }
  });

  describe('chunk', () => {
    it('divides evenly', async () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
      const expected = CPUTensor.chunk(CPUTensor.fromArray(data, [6, 2]), 3);
      const actual = WebGPUTensor.chunk(
        WebGPUTensor.fromArray(data, [6, 2]),
        3
      );
      assert.equal(actual.length, expected.length);
      for (let i = 0; i < expected.length; i++) {
        assert.deepEqual(actual[i].shape, expected[i].shape);
        assert.deepEqual(
          await actual[i].toArrayAsync(),
          await expected[i].toArrayAsync()
        );
      }
    });

    it('divides with a remainder along dim 1', async () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const expected = CPUTensor.chunk(CPUTensor.fromArray(data, [2, 5]), 3, 1);
      const actual = WebGPUTensor.chunk(
        WebGPUTensor.fromArray(data, [2, 5]),
        3,
        1
      );
      assert.equal(actual.length, expected.length);
      for (let i = 0; i < expected.length; i++) {
        assert.deepEqual(actual[i].shape, expected[i].shape);
        assert.deepEqual(
          await actual[i].toArrayAsync(),
          await expected[i].toArrayAsync()
        );
      }
    });
  });

  describe('max / min / argmax / argmin', () => {
    // 同値を含めて、同着のときに先頭のインデックスが返ることを確認する
    const data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9];
    const shape = [2, 2, 3];

    const compare = async (
      actual: WebGPUTensor,
      expected: CPUTensor,
      message: string
    ) => {
      assert.deepEqual(actual.shape, expected.shape, `${message}: shape`);
      assert.equal(actual.dtype, expected.dtype, `${message}: dtype`);
      assert.deepEqual(
        await actual.toArrayAsync(),
        await expected.toArrayAsync(),
        message
      );
    };

    it('max / min over all elements', async () => {
      const g = WebGPUTensor.fromArray(data, shape);
      const c = CPUTensor.fromArray(data, shape);
      await compare(WebGPUTensor.max(g), CPUTensor.max(c), 'max');
      await compare(WebGPUTensor.min(g), CPUTensor.min(c), 'min');
    });

    it('argmax / argmin over all elements', async () => {
      const g = WebGPUTensor.fromArray(data, shape);
      const c = CPUTensor.fromArray(data, shape);
      await compare(WebGPUTensor.argmax(g), CPUTensor.argmax(c), 'argmax');
      await compare(WebGPUTensor.argmin(g), CPUTensor.argmin(c), 'argmin');
    });

    for (const dim of [0, 1, 2, -1]) {
      for (const keepdim of [false, true]) {
        it(`max / min along dim=${dim} keepdim=${keepdim}`, async () => {
          const g = WebGPUTensor.fromArray(data, shape);
          const c = CPUTensor.fromArray(data, shape);
          const [gv, gi] = WebGPUTensor.max(g, dim, keepdim);
          const [cv, ci] = CPUTensor.max(c, dim, keepdim);
          await compare(gv, cv, 'max values');
          await compare(gi, ci, 'max indices');
          const [gv2, gi2] = WebGPUTensor.min(g, dim, keepdim);
          const [cv2, ci2] = CPUTensor.min(c, dim, keepdim);
          await compare(gv2, cv2, 'min values');
          await compare(gi2, ci2, 'min indices');
        });

        it(`argmax / argmin along dim=${dim} keepdim=${keepdim}`, async () => {
          const g = WebGPUTensor.fromArray(data, shape);
          const c = CPUTensor.fromArray(data, shape);
          await compare(
            WebGPUTensor.argmax(g, dim, keepdim),
            CPUTensor.argmax(c, dim, keepdim),
            'argmax'
          );
          await compare(
            WebGPUTensor.argmin(g, dim, keepdim),
            CPUTensor.argmin(c, dim, keepdim),
            'argmin'
          );
        });
      }
    }
  });
});

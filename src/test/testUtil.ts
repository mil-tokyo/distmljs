import { assert } from 'chai';

/**
 * assert.instanceOfの型定義はpublicなコンストラクタを持つクラスしか受け付けないが、
 * Tensorの各実装クラスはprivateコンストラクタを持つ。実行時にはinstanceof演算子で
 * 判定されるだけなので、キャストしてchaiに渡す。
 */
export function assertInstanceOf(
  value: unknown,
  ctor: Function, // eslint-disable-line @typescript-eslint/ban-types
  message?: string
): void {
  assert.instanceOf(value, ctor as new (...args: never[]) => unknown, message);
}

export function arrayNearlyEqual(
  a: ReadonlyArray<number>,
  b: ReadonlyArray<number>,
  message?: string | null,
  atol = 1e-3,
  rtol = 1e-2
): void {
  if (a.length !== b.length) {
    assert.fail(
      `${a.length}`,
      `${b.length}`,
      `${message}: Array length does not match: ${a.length} !== ${b.length}`
    );
  }
  for (let i = 0; i < a.length; i++) {
    const va = a[i];
    const vb = b[i];
    const diff = Math.abs(va - vb);
    // diff > atol + rtol * Math.abs(vb) だとNaNが検出できない
    if (!(diff <= atol + rtol * Math.abs(vb))) {
      assert.fail(
        `${va}`,
        `${vb}`,
        `${message}: Value[${i}] not equal: ${va} !== ${vb}`
      );
    }
  }
}

import { assert } from 'chai';

export function arrayNearlyEqual(
  a: ReadonlyArray<number>,
  b: ReadonlyArray<number>,
  atol = 1e-3,
  rtol = 1e-2
): void {
  if (a.length !== b.length) {
    assert.fail(`Array length does not match: ${a.length} !== ${b.length}`);
  }
  for (let i = 0; i < a.length; i++) {
    const va = a[i];
    const vb = b[i];
    const diff = Math.abs(va - vb);
    // diff > atol + rtol * Math.abs(vb) だとNaNが検出できない
    if (!(diff <= atol + rtol * Math.abs(vb))) {
      assert.fail(`Value[${i}] not equal: ${va} !== ${vb}`);
    }
  }
}

import Long from 'long';

const longPositive32BitMax = new Long(0x7fffffff, 0),
  longPositive32BitMin = new Long(0x80000000, 0xffffffff);

// 符号付きLongを丸めて、-2^31から2^31-1の範囲のnumberを返す
export function clipLong(v: Long): number {
  // Long(0xfffffff6, 0xffffffff) => -10
  if (v.lessThan(longPositive32BitMin)) {
    return -0x80000000;
  } else if (v.greaterThan(longPositive32BitMax)) {
    return 0x7fffffff;
  }
  return v.toNumber();
}

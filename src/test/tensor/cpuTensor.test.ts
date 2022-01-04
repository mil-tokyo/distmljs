import { assert } from 'chai';
import { CPUTensor } from '../../cpuTensor';

describe('basic', () => {
  it('computes size', () => {
    const t = new CPUTensor([3, 4]);
    assert.equal(t.size, 12);
  });

  it('computes size of scalar', () => {
    const t = new CPUTensor([]);
    assert.equal(t.size, 1);
  });
});

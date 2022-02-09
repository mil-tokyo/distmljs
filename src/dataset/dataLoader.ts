import { CPUTensor } from '../tensor';
import { Dataset } from './dataset';

export interface DataLoaderOptions {
  batchSize: number;
  shuffle?: boolean;
}

export class DataLoader {
  length: number;

  constructor(public dataset: Dataset, public options: DataLoaderOptions) {
    this.length = Math.floor(dataset.length / options.batchSize);
    if (options.shuffle) {
      console.warn('Dataset shuffle is not yet implemented');
    }
  }

  async *[Symbol.asyncIterator]() {
    // usage: for await (const [images, labels] of this)
    for (let batch = 0; batch < this.length; batch++) {
      const bottom = batch * this.options.batchSize;
      const top = (batch + 1) * this.options.batchSize;
      const elems: CPUTensor[][] = [];
      for (let sample = bottom; sample < top; sample++) {
        const elem = await this.dataset.getAsync(sample);
        elems.push(elem);
      }
      const allStacked: CPUTensor[] = [];
      for (let obj = 0; obj < elems[0].length; obj++) {
        allStacked.push(this.stack(elems, obj));
      }
      yield allStacked;
    }
  }

  private stack(elems: CPUTensor[][], objIndex: number): CPUTensor {
    const first = elems[0][objIndex];
    const oneSize = first.size;
    const allShape = [elems.length, ...first.shape];
    const stacked = CPUTensor.zeros(allShape, first.dtype);
    const sdata = stacked.buffer.data;
    for (let i = 0; i < elems.length; i++) {
      const elem = elems[i][objIndex];
      if (elem.size !== oneSize) {
        throw new Error('Size of batch elements mismatch');
      }
      const edata = elem.buffer.data;
      sdata.set(edata, i * oneSize);
    }
    return stacked;
  }
}

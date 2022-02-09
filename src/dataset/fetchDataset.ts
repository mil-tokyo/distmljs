import { Dataset } from '.';
import { CPUTensor, TensorDeserializer } from '../tensor';

export class FetchDataset implements Dataset {
  length!: number;
  data!: CPUTensor;
  targets!: CPUTensor;

  constructor(public url: string) {}

  async load(): Promise<void> {
    const td = new TensorDeserializer();
    const tensors = await td.fromHTTP(this.url);
    const data = tensors.get('data');
    const targets = tensors.get('targets');
    if (!data || !targets) {
      throw new Error('data, targets not in the dataset');
    }
    this.data = data;
    this.targets = targets;
    this.length = data.shape[0];
  }

  async getAsync(idx: number): Promise<CPUTensor[]> {
    return [this.data.gets(idx), this.targets.gets(idx)];
  }
}

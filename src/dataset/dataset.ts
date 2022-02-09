import { CPUTensor } from '../tensor';

export interface Dataset {
  length: number;
  getAsync(idx: number): Promise<CPUTensor[]>;
}

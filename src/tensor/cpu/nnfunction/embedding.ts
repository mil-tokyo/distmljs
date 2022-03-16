import { CPUTensor } from '../cpuTensor';

export function embedding_cpu(x: CPUTensor, weight: CPUTensor): CPUTensor {
  const [numEmbeddings, embeddingDim] = weight.shape;
  const xsize = x.size;
  const output = CPUTensor.zeros([...x.shape, embeddingDim]);
  const dx = x.getBuffer().data;
  const dw = weight.getBuffer().data;
  const dy = output.getBuffer().data;

  for (let i = 0; i < xsize; i++) {
    const idx = dx[i];
    if (idx < 0 || idx >= numEmbeddings) {
      throw new Error(`embedding: index ${idx} out of range`);
    }
    for (let j = 0; j < embeddingDim; j++) {
      dy[i * embeddingDim + j] = dw[idx * embeddingDim + j];
    }
  }
  return output;
}

export function embedding_backprop_cpu(
  x: CPUTensor,
  gy: CPUTensor,
  numEmbeddings: number,
  embeddingDim: number
): CPUTensor {
  const xsize = x.size;
  const output = CPUTensor.zeros([numEmbeddings, embeddingDim]);
  const dx = x.getBuffer().data;
  const dgy = gy.getBuffer().data;
  const dgx = output.getBuffer().data;

  for (let i = 0; i < xsize; i++) {
    const idx = dx[i];
    for (let j = 0; j < embeddingDim; j++) {
      dgx[idx * embeddingDim + j] += dgy[i * embeddingDim + j];
    }
  }

  return output;
}

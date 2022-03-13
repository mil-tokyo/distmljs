import * as K from 'kakiage';
import CPUTensor = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;

export function log(message: string, time = false): void {
  const div = document.getElementById('result');
  const elem = document.createElement('div');
  let m = message;
  if (time) {
    m = `${performance.now() | 0}ms: ${m}`;
  }
  elem.innerText = m;
  div?.appendChild(elem);
}

export function wait() {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 10);
  });
}

export function generateSquareSubsquentMask(sz: number): CPUTensor {
  return CPUTensor.triu(CPUTensor.full([sz, sz], -Infinity), 1);
}

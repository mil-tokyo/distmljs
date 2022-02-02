import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import Random = K.math.Random;

function print(message: string, time = false): void {
  const div = document.getElementById('result');
  const elem = document.createElement('div');
  let m = message;
  if (time) {
    m = `${performance.now() | 0}ms: ${m}`;
  }
  elem.innerText = m;
  div?.appendChild(elem);
}

async function run() {
  print('start');
  await K.tensor.initializeNNWebGLContext();
  print('init ok');
  try {
    const x = K.tensor.WebGLTensor.fromArray([1.0, -1.0]);
    print('fa ok');
    const y = K.tensor.WebGLTensor.exp(x);
    print('exp ok');
    print(`${await y.toArrayAsync()}`);
  } catch (error) {
    print(`Error: ${error}`);
  }
}

async function run2() {
  print('start2darray');
  try {
    const x = K.tensor.WebGLTensor.empty([2, 2, 2], undefined, undefined, {
      internalFormat: WebGL2RenderingContext.R16F,
      format: WebGL2RenderingContext.RED,
      type: WebGL2RenderingContext.HALF_FLOAT,
      dim: '2DArray',
      width: 2,
      height: 2,
      depth: 2,
    });
    x.setArray([1, 2, 3, 4, 5, 6, 7, 8]);
    print('fa ok');
    const y = K.tensor.WebGLTensor.exp(x);
    print('exp ok');
    print(`${await y.toArrayAsync()}`);
  } catch (error) {
    console.error(error);
    print(`Error: ${error}`);
  }
}
window.addEventListener('load', async () => {
  await run();
  await run2();
});

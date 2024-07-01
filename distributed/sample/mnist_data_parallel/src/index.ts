import { throttle } from 'lodash';
import * as K from 'distmljs';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;
import F = K.nn.functions;
import L = K.nn.layers;

let ws: WebSocket;
let backend: K.Backend = 'webgl';

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

class NetMLP extends K.nn.core.Layer {
  fc1: K.nn.layers.Linear;
  fc2: K.nn.layers.Linear;

  constructor(inputShape: number[], nClasses: number) {
    super();

    this.fc1 = new K.nn.layers.Linear(
      inputShape[0] * inputShape[1] * inputShape[2],
      32
    );
    this.fc2 = new K.nn.layers.Linear(32, nClasses);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await K.nn.functions.flatten(y);
    y = await this.fc1.c(y);
    y = await K.nn.functions.relu(y);
    y = await this.fc2.c(y);
    return [y];
  }
}

class NetConv extends K.nn.core.Layer {
  conv1: K.nn.layers.Conv2d;
  conv2: K.nn.layers.Conv2d;
  conv3: K.nn.layers.Conv2d;
  fc1: K.nn.layers.Linear;

  constructor(inputShape: number[], nClasses: number) {
    super();

    this.conv1 = new K.nn.layers.Conv2d(inputShape[0], 8, 3, {});
    this.conv2 = new K.nn.layers.Conv2d(8, 16, 3, {});
    this.conv3 = new K.nn.layers.Conv2d(16, 32, 3, {});
    this.fc1 = new K.nn.layers.Linear(32, nClasses);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await K.nn.functions.relu(y);
    y = await K.nn.functions.max_pool2d(y, { kernelSize: 2 });
    y = await this.conv2.c(y);
    y = await K.nn.functions.relu(y);
    y = await K.nn.functions.max_pool2d(y, { kernelSize: 2 });
    y = await this.conv3.c(y);
    y = await K.nn.functions.relu(y);
    y = await K.nn.functions.adaptive_avg_pool2d(y, 1);
    y = await K.nn.functions.flatten(y);
    y = await this.fc1.c(y);
    return [y];
  }
}

class BasicBlock extends K.nn.core.Layer {
  conv1: L.Conv2d;
  bn1: L.BatchNorm;
  conv2: L.Conv2d;
  bn2: L.BatchNorm;
  downsampleConv?: L.Conv2d;
  downsampleBN?: L.BatchNorm;

  constructor(
    inPlanes: number,
    planes: number,
    stride: number,
    downsample: boolean
  ) {
    super();
    this.conv1 = new L.Conv2d(inPlanes, planes, 3, {
      stride: stride,
      padding: 1,
      bias: false,
    });
    this.bn1 = new L.BatchNorm(planes, {});
    this.conv2 = new L.Conv2d(planes, planes, 3, { padding: 1, bias: false });
    this.bn2 = new L.BatchNorm(planes, {});
    if (downsample) {
      this.downsampleConv = new L.Conv2d(inPlanes, planes, 1, {
        stride: stride,
        bias: false,
      });
      this.downsampleBN = new L.BatchNorm(planes, {});
    }
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    const x = inputs[0];
    let y = x;
    y = await this.conv1.c(y);
    y = await this.bn1.c(y);
    y = await F.relu(y);
    y = await this.conv2.c(y);
    y = await this.bn2.c(y);
    let ds: Variable;
    if (this.downsampleConv && this.downsampleBN) {
      ds = await this.downsampleConv.c(x);
      ds = await this.downsampleBN.c(ds);
    } else {
      ds = x;
    }
    y = await F.add(ds, y);
    y = await F.relu(y);
    return [y];
  }
}

export class ResNet18 extends K.nn.core.Layer {
  conv1: L.Conv2d;
  bn1: L.BatchNorm;
  fc: L.Linear;
  layer1: L.Sequential;
  layer2: L.Sequential;
  layer3: L.Sequential;
  layer4: L.Sequential;

  constructor(inputShape: number[], nClass: number) {
    super();
    this.conv1 = new L.Conv2d(inputShape[0], 64, 7, {
      stride: 2,
      padding: 3,
      bias: false,
    });
    this.bn1 = new L.BatchNorm(64, {});
    this.layer1 = this.makeLayer(64, 64, 2, 1);
    this.layer2 = this.makeLayer(64, 128, 2, 2);
    this.layer3 = this.makeLayer(128, 256, 2, 2);
    this.layer4 = this.makeLayer(256, 512, 2, 2);
    this.fc = new L.Linear(512, nClass);
    this.initConvWeights();
  }

  private makeLayer(
    inPlanes: number,
    planes: number,
    blocks: number,
    stride: number
  ): L.Sequential {
    const layers: K.nn.core.Layer[] = [];
    layers.push(new BasicBlock(inPlanes, planes, stride, stride !== 1));
    for (let i = 1; i < blocks; i++) {
      layers.push(new BasicBlock(planes, planes, 1, false));
    }
    return new L.Sequential(...layers);
  }

  private initConvWeights() {
    for (const param of this.parameters()) {
      if (param.data.ndim === 4) {
        this.initConvWeight(param);
      }
    }
  }

  private initConvWeight(param: K.nn.core.Parameter) {
    // special initialization as in PyTorch's sample
    const rnd = K.math.Random.getDefault();
    const s = param.data.shape;
    const fanOut = s[0] * s[2] * s[3];
    const gain = Math.SQRT2;
    const std = gain / Math.sqrt(fanOut);
    const length = param.data.size;
    const vec = rnd.normal(length);
    for (let i = 0; i < length; i++) {
      vec[i] = vec[i] * std;
    }
    (param.data as K.tensor.CPUTensor).setArray(vec);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await this.bn1.c(y);
    y = await F.relu(y);
    y = await F.max_pool2d(y, { kernelSize: 3, stride: 2, padding: 1 });
    y = await this.layer1.c(y);
    y = await this.layer2.c(y);
    y = await this.layer3.c(y);
    y = await this.layer4.c(y);
    y = await F.adaptive_avg_pool2d(y, 1);
    y = await F.flatten(y);
    y = await this.fc.c(y);
    return [y];
  }
}

function makeModel(
  name: string,
  inputShape: number[],
  nClasses: number
): K.nn.core.Layer {
  switch (name) {
    case 'mlp':
      return new NetMLP(inputShape, nClasses);
    case 'conv':
      return new NetConv(inputShape, nClasses);
    case 'resnet18':
      return new ResNet18(inputShape, nClasses);
    default:
      throw new Error(`Unknown model ${name}`);
  }
}

const writeState = throttle((message: string) => {
  document.getElementById('state')!.innerText = message;
}, 1000);

const writeBatchInfo = throttle((processedBatches: number, lastLoss: number, batchSize: number) => {
  document.getElementById('table-batches')!.innerText = processedBatches.toString();
  document.getElementById('table-loss')!.innerText = lastLoss.toString();
  document.getElementById('table-batchsize')!.innerText = batchSize.toString();
}, 1000);

async function sendBlob(itemId: string, data: Uint8Array): Promise<void> {
  const blob = new Blob([data]);
  const f = await fetch(`/distmljs/blob/${itemId}`, {
    method: 'PUT',
    body: blob,
    headers: { 'Content-Type': 'application/octet-stream' },
  });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  await f.text(); // avoid memory leak
}

async function recvBlob(itemId: string): Promise<Uint8Array> {
  const f = await fetch(`/distmljs/blob/${itemId}`, { method: 'GET' });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await f.arrayBuffer();
  return new Uint8Array(resp);
}

let model: K.nn.core.Layer;

let totalBatches = 0;

interface computePerformance {
  receiveInput: number;
  computeGrad: number;
  sendOutput: number;
}

async function compute(msg: { weight: string; dataset: string; grad: string }) {
  const time1 = performance.now();
  const [weightBlob, datasetBlob] = await Promise.all([recvBlob(msg.weight), recvBlob(msg.dataset)]);
  const time2 = performance.now();
  const weights = new TensorDeserializer().deserialize(weightBlob);
  for (const { name, parameter } of model.parametersWithName(true, false)) {
    parameter.data = await nonNull(weights.get(name)).to(backend);
    parameter.cleargrad();
  }
  const dataset = new TensorDeserializer().deserialize(datasetBlob);
  const image = await nonNull(dataset.get('image')).to(backend);
  const label = await nonNull(dataset.get('label')).to(backend);
  const y = await model.c(new K.nn.Variable(image));
  const labelV = new K.nn.Variable(label);
  const loss = await K.nn.functions.softmaxCrossEntropy(y, labelV);
  const lossValue = (await loss.data.to('cpu')).get(0);
  console.log(`loss: ${lossValue}`);
  await loss.backward();
  const grads = new Map<string, T>();
  for (const { name, parameter } of model.parametersWithName(true, false)) {
    if (parameter.optimizable) {
      grads.set(name, await nonNull(parameter.grad).data.to('cpu'));
    } else {
      // statistics of BN (runningMean, runningVar, numBatchesTracked)
      grads.set(name, await nonNull(parameter.data).to('cpu'));
    }
  }
  const time3 = performance.now();
  await sendBlob(msg.grad, new TensorSerializer().serialize(grads));
  const time4 = performance.now();
  totalBatches += 1;
  writeBatchInfo(totalBatches, lossValue, y.data.shape[0]);
  return {
    receiveInput: time2 - time1,
    computeGrad: time3 - time2,
    sendOutput: time4 - time3,
  };
}

async function run() {
  writeState('Connecting to distributed training server...');
  ws = new WebSocket(
    (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
    window.location.host +
    '/distmljs/ws'
  );
  ws.onopen = () => {
    writeState('Connected to server');
  };
  ws.onclose = () => {
    writeState('Disconnected from server');
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (!model) {
      model = makeModel(msg.model, msg.inputShape, msg.nClasses);
      await model.to(backend);
    }
    let performance: computePerformance | undefined;
    await K.tidy(async () => {
      performance = await compute(msg);
      return [];
    });
    ws.send(JSON.stringify({ performance }));
  };
}

window.addEventListener('load', async () => {
  backend = (new URLSearchParams(window.location.search).get('backend') ||
    'webgl') as K.Backend;
  if (backend === 'webgl') {
    await K.tensor.initializeNNWebGLContext();
  }
  await run();
});

import { throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;

let ws: WebSocket;
let backend: K.Backend = "webgl";

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

class NetMLP extends K.nn.core.Layer {
  fc1: K.nn.layers.Linear;
  fc2: K.nn.layers.Linear;

  constructor(inFeatures=784, hidden=32, outFeatures=10) {
    super();
    
    this.fc1 = new K.nn.layers.Linear(inFeatures, hidden);
    this.fc2 = new K.nn.layers.Linear(hidden, outFeatures);
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
  fc1: K.nn.layers.Linear;

  constructor() {
    super();

    this.conv1 = new K.nn.layers.Conv2d(1, 8, 3, {});
    this.conv2 = new K.nn.layers.Conv2d(8, 8, 3, {});
    this.fc1 = new K.nn.layers.Linear(200, 10);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await K.nn.functions.relu(y);
    y = await K.nn.functions.max_pool2d(y, {kernelSize: 2});
    y = await this.conv2.c(y);
    y = await K.nn.functions.relu(y);
    y = await K.nn.functions.max_pool2d(y, {kernelSize: 2});    y = await K.nn.functions.flatten(y);
    y = await this.fc1.c(y);
    return [y];
  }
}

function makeModel(name: string): K.nn.core.Layer {
  switch (name) {
    case "mlp":
      return new NetMLP();
    case "conv":
      return new NetConv();
    default:
      throw new Error(`Unknown model ${name}`);
  }
}

function writeLog(message: string) {
  document.getElementById("messages")!.innerText += message + "\n";
}

const writeState = throttle((message: string) => {
  document.getElementById('state')!.innerText = message;
}, 1000);

async function sendBlob(itemId: string, data: Uint8Array): Promise<void> {
  const blob = new Blob([data]);
  const f = await fetch(`/kakiage/blob/${itemId}`, {
    method: 'PUT',
    body: blob,
    headers: { 'Content-Type': 'application/octet-stream' },
  });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
}

async function recvBlob(itemId: string): Promise<Uint8Array> {
  const f = await fetch(`/kakiage/blob/${itemId}`, { method: 'GET' });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await f.arrayBuffer();
  return new Uint8Array(resp);
}

let model: K.nn.core.Layer;

let totalBatches = 0;

async function compute(msg: { weight: string; dataset: string; grad: string }) {
  const weights = new TensorDeserializer().deserialize(
    await recvBlob(msg.weight)
  );
  for (const { name, parameter } of model.parametersWithName()) {
    parameter.data = await nonNull(weights.get(name)).to(backend);
    parameter.cleargrad();
  }
  const dataset = new TensorDeserializer().deserialize(
    await recvBlob(msg.dataset)
  );
  const image = await nonNull(dataset.get('image')).to(backend);
  const label = await nonNull(dataset.get('label')).to(backend);
  const y = await model.c(new K.nn.Variable(image));
  const labelV = new K.nn.Variable(label);
  const loss = await K.nn.functions.softmaxCrossEntropy(y, labelV);
  const lossValue = (await loss.data.to("cpu")).get(0);
  console.log(`loss: ${lossValue}`);
  await loss.backward();
  const grads = new Map<string, T>();
  for (const { name, parameter } of model.parametersWithName()) {
    grads.set(name, await nonNull(parameter.grad).data.to("cpu"));
  }
  await sendBlob(msg.grad, new TensorSerializer().serialize(grads));
  totalBatches += 1;
  writeState(
    `total batch: ${totalBatches}, last loss: ${lossValue}, last batch size: ${y.data.shape[0]}`
  );
}

async function run() {
  writeState('Connecting');
  ws = new WebSocket(
    (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
      window.location.host +
      '/kakiage/ws'
  );
  ws.onopen = () => {
    writeState('Connected to WS server');
  };
  ws.onclose = () => {
    writeState('Disconnected from WS server');
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (!model) {
      model = makeModel(msg.model);
      await model.to(backend);
    }
    await K.tidy(async () => {
      await compute(msg);
      return [];
    });
    ws.send(JSON.stringify({}));
  };
}

window.addEventListener('load', async () => {
  backend = (new URLSearchParams(window.location.search).get("backend") || "webgl") as K.Backend;
  writeLog(`backend: ${backend}`);
  if (backend === "webgl") {
    await K.tensor.initializeNNWebGLContext();
  }
  await run();
});

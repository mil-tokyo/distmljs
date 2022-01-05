import { throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;

let ws: WebSocket;

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

class Net extends K.nn.core.Layer {
  fc1: K.nn.layers.Linear;
  fc2: K.nn.layers.Linear;

  constructor(inFeatures: number, hidden: number, outFeatures: number) {
    super();
    this.fc1 = new K.nn.layers.Linear(inFeatures, hidden);
    this.fc2 = new K.nn.layers.Linear(hidden, outFeatures);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.fc1.c(y);
    y = await K.nn.functions.relu(y);
    y = await this.fc2.c(y);
    return [y];
  }
}

function writeLog(message: string) {
  //document.getElementById("messages")!.innerText += message + "\n";
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

const model = new Net(784, 32, 10);

let totalBatches = 0;

async function compute(msg: { weight: string; dataset: string; grad: string }) {
  const weights = new TensorDeserializer().deserialize(
    await recvBlob(msg.weight)
  );
  for (const { name, parameter } of model.parametersWithName()) {
    parameter.data = nonNull(weights.get(name));
    parameter.cleargrad();
  }
  const dataset = new TensorDeserializer().deserialize(
    await recvBlob(msg.dataset)
  );
  const y = await model.c(new K.nn.Variable(nonNull(dataset.get('image'))));
  const label = new K.nn.Variable(nonNull(dataset.get('label')));
  const loss = await K.nn.functions.softmaxCrossEntropy(y, label);
  const lossValue = (loss.data as T).get(0);
  console.log(`loss: ${lossValue}`);
  await loss.backward();
  const grads = new Map<string, T>();
  for (const { name, parameter } of model.parametersWithName()) {
    grads.set(name, nonNull(parameter.grad).data as T);
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
    await compute(msg);
    ws.send(JSON.stringify({}));
  };
}

window.addEventListener('load', () => {
  run();
});

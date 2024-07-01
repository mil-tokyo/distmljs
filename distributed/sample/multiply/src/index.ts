import * as K from 'distmljs';
import T = K.tensor.CPUTensor;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;

let ws: WebSocket;

function writeLog(message: string) {
  document.getElementById("messages")!.innerText += message + "\n";
}

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
}

async function recvBlob(itemId: string): Promise<Uint8Array> {
  const f = await fetch(`/distmljs/blob/${itemId}`, { method: 'GET' });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await f.arrayBuffer();
  return new Uint8Array(resp);
}

async function compute(msg: { src: string; dst: string }) {
  writeLog("receiving src");
  const srcBlob = await recvBlob(msg.src);
  writeLog("deserializing src");
  const srcTensor = (new TensorDeserializer()).deserialize(srcBlob).get("src")!;
  const dstTensor = T.mul(srcTensor, T.fromArray([2]));
  writeLog("sending dst");
  await sendBlob(msg.dst, (new TensorSerializer()).serialize({ "dst": dstTensor }));
  writeLog("sent dst");
}

async function run() {
  writeLog("Connecting");
  ws = new WebSocket(
    (window.location.protocol === "https:" ? "wss://" : "ws://") +
    window.location.host +
    "/distmljs/ws"
  );
  ws.onopen = () => {
    writeLog("Connected to WS server");
  };
  ws.onclose = () => {
    writeLog("Disconnected from WS server");
  };
  ws.onmessage = async (ev) => {
    writeLog(`Message: ${ev.data}`);
    const msg = JSON.parse(ev.data);
    await compute(msg);
    ws.send(JSON.stringify({}));
    writeLog("sent complete message");
  };
}

window.addEventListener("load", () => {
  run();
});

let ws: WebSocket;

function writeLog(message: string) {
  document.getElementById("messages")!.innerText += message + "\n";
}

let totalPoints = 0;

function addPoint(points: number) {
  totalPoints += points;
  document.getElementById("n_processed")!.innerText = totalPoints.toString();
}

async function compute(msg: { run_per_ticket: number }): Promise<{ run: number; hit: number }> {
  let hit = 0;
  let run = msg.run_per_ticket;
  for (let i = 0; i < run; i++) {
    let x = Math.random();
    let y = Math.random();
    let dist = x * x + y * y;
    if (dist < 1) {
      hit++;
    }
  }
  addPoint(run);
  return { run, hit };
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
    const msg = JSON.parse(ev.data);
    const ret = await compute(msg);
    ws.send(JSON.stringify(ret));
  };
}

window.addEventListener("load", () => {
  run();
});
